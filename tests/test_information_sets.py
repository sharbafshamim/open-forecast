"""Leakage audit: every model family sees only the data its information set declares.

For a forecast origin t and horizon H the declared data are
    A   calendar of t+H, weather up to t+H (observed past + declared proxy)
    B   A + load observed in [t-23, t]
    B+  A + all load observed up to t
The tests perturb everything OUTSIDE that window (future load, weather after
t+H, and for A / B the load the set does not declare) and assert that the
inputs a model receives for origin t do not change. They cover the tabular
features (trees), the sequence tensors (deep models) and the contexts and
covariates handed to the foundation models.
"""

import numpy as np
import pandas as pd
import pytest

from open_forecast import foundation
from open_forecast.contracts import Split
from open_forecast.features import build_sequence_partitions, build_tabular_frame

H = 24
ZONES = ["Z1", "Z2"]
WEATHER = ["Z1_Dry_Bulb", "Z2_Dry_Bulb"]
MODES = ["weather_time", "weather_time_history", "weather_time_history_weekly"]
LAGS, ROLLS = [0, 1, 2, 3, 6, 12, 23], [3, 24]


def _hourly(days=30, seed=3):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    data = {z: 100 + rng.normal(size=len(index)) for z in ZONES}
    data.update({w: rng.normal(size=len(index)) for w in WEATHER})
    return pd.DataFrame(data, index=index)


def _perturbations(hourly, t, mode):
    """Copies of the data changed only outside the window declared for (t, mode)."""
    out = {}
    later_load = hourly.copy()
    later_load.loc[hourly.index > t, ZONES] += 50.0
    out["load after t"] = later_load
    late_weather = hourly.copy()
    late_weather.loc[hourly.index > t + pd.Timedelta(hours=H), WEATHER] += 50.0
    out["weather after t+H"] = late_weather
    if mode == "weather_time":
        any_load = hourly.copy()
        any_load[ZONES] += 50.0
        out["all load"] = any_load
    if mode == "weather_time_history":
        old_load = hourly.copy()
        old_load.loc[hourly.index < t - pd.Timedelta(hours=23), ZONES] += 50.0
        out["load before t-23"] = old_load
    return out


def _tabular_row(hourly, mode, t):
    x, _ = build_tabular_frame(hourly, ZONES, WEATHER, mode, H, LAGS, ROLLS,
                               weather_path_stats=True)
    return x.loc[t].to_numpy(dtype=float)


@pytest.mark.parametrize("mode", MODES)
def test_tabular_features_use_only_declared_data(mode):
    hourly = _hourly()
    t = hourly.index[24 * 20]
    reference = _tabular_row(hourly, mode, t)
    for name, changed in _perturbations(hourly, t, mode).items():
        assert np.allclose(_tabular_row(changed, mode, t), reference), f"{mode}: depends on {name}"


def _sequence_sample(hourly, mode, t):
    index = hourly.index
    split = Split(train=index[200:300], validation=index[330:380], test=pd.DatetimeIndex([t]),
                  horizon_hours=H, validation_embargoed_origins=H, test_embargoed_origins=H)
    parts = build_sequence_partitions(hourly, ZONES, WEATHER, mode, H, 24, split)
    return parts["test"].x[0]


@pytest.mark.parametrize("mode", MODES)
def test_sequence_inputs_use_only_declared_data(mode):
    hourly = _hourly()
    t = hourly.index[24 * 20]
    reference = _sequence_sample(hourly, mode, t)
    for name, changed in _perturbations(hourly, t, mode).items():
        assert np.allclose(_sequence_sample(changed, mode, t), reference), f"{mode}: depends on {name}"


class _Recorder:
    """Fake foundation pipeline that records its inputs and returns zeros."""

    def __init__(self, covariates):
        self.covariates = covariates
        self.inputs = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.inputs.extend(inputs)
        zeros = np.zeros(prediction_length)
        if self.covariates:
            return [zeros[None, :, None] for _ in inputs], None
        return np.stack([zeros[:, None] for _ in inputs]), None


@pytest.mark.parametrize("model", ["chronos_bolt_small", "chronos2"])
def test_foundation_inputs_use_only_set_b_plus_data(model, monkeypatch):
    """Chronos contexts are load up to t (set B+) and weather up to t+H."""
    recorders = {}

    def factory(name, device):
        recorders[name] = _Recorder(name in foundation.COVARIATE_MODELS)
        return recorders[name]

    monkeypatch.setattr(foundation, "pipeline_factory", factory)

    def inputs_for(frame):
        monkeypatch.setattr(foundation, "_PIPELINES", {})
        t = frame.index[24 * 20]
        foundation.predict_load(model, frame, pd.DatetimeIndex([t]), ZONES, WEATHER, H,
                                {"context_hours": 300})
        return recorders[model].inputs

    hourly = _hourly()
    t = hourly.index[24 * 20]
    reference = inputs_for(hourly)
    for name, changed in _perturbations(hourly, t, "weather_time_history_weekly").items():
        got = inputs_for(changed)
        for a, b in zip(reference, got):
            if isinstance(a, dict):
                assert np.allclose(a["target"], b["target"]), name
                for key in ("past_covariates", "future_covariates"):
                    for column in a[key]:
                        assert np.allclose(a[key][column], b[key][column]), f"{name}: {key} {column}"
            else:
                assert np.allclose(np.asarray(a), np.asarray(b)), name
