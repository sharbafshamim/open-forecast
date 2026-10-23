"""Tests for price forecasting, rolling-origin folds and foundation-model plumbing.

None of these tests needs torch or downloaded weights: a deterministic fake
pipeline that returns "the value one week earlier" must reproduce the
seasonal-naive forecast exactly, which proves context/horizon alignment.
"""

import numpy as np
import pandas as pd
import pytest

from open_forecast import foundation
from open_forecast.price import (
    AsinhScaler,
    PriceData,
    build_design,
    multivariate_dm,
    naive_forecast,
    rolling_forecast,
)
from open_forecast.rolling import make_fold_builder


class WeeklyFake:
    """predict_quantiles returning y(context_end + k - 168) for k = 1..H."""

    def __init__(self, covariates: bool):
        self.covariates = covariates
        self.calls = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        out = []
        for item in inputs:
            target = np.asarray(item["target"] if self.covariates else item, dtype=float)
            self.calls.append(item)
            path = np.array([target[-168 + k - 1] for k in range(1, prediction_length + 1)])
            out.append(path[None, :, None] if self.covariates else path[:, None])
        return (out if self.covariates else np.stack(out)), None


@pytest.fixture
def fake_pipelines(monkeypatch):
    fakes = {}

    def factory(model, device):
        fakes[model] = WeeklyFake(model in foundation.COVARIATE_MODELS)
        return fakes[model]

    monkeypatch.setattr(foundation, "pipeline_factory", factory)
    monkeypatch.setattr(foundation, "_PIPELINES", {})
    return fakes


def _hourly(days=60, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    base = 50 + 10 * np.sin(2 * np.pi * index.hour / 24) + rng.normal(0, 3, len(index))
    return pd.DataFrame({"Price": base, "Temp": rng.normal(10, 5, len(index)),
                         "Load": rng.normal(100, 5, len(index))}, index=index)


def test_price_design_uses_only_information_known_on_d_minus_1():
    hourly = _hourly()
    data = PriceData(hourly, {"Temp": (0,), "Load": (1, 7)}, {})
    x, y = build_design(data, "price_exogenous")
    day = x.index[10]
    prev = day - pd.Timedelta(days=1)
    assert np.allclose(x.loc[day, [f"price_d1_h{h}" for h in range(24)]], y.loc[prev])
    assert not any(c.startswith("Load_d0") for c in x.columns)       # day-D load never used
    assert any(c.startswith("Temp_d0") for c in x.columns)            # declared proxy is
    assert not any(c.startswith("price_d0") for c in x.columns)


def test_naive_uses_week_for_mon_sat_sun():
    hourly = _hourly()
    _, y = build_design(PriceData(hourly, {}, {}), "price_only")
    monday = y.index[y.index.dayofweek == 0][2]
    tuesday = monday + pd.Timedelta(days=1)
    assert np.allclose(naive_forecast(y, monday), y.loc[monday - pd.Timedelta(days=7)])
    assert np.allclose(naive_forecast(y, tuesday), y.loc[monday])


def test_rolling_lear_never_trains_on_test_block(monkeypatch):
    hourly = _hourly(days=90)
    x, y = build_design(PriceData(hourly, {}, {}), "price_only")
    seen = []
    import open_forecast.price as price

    def spy(model, x_train, y_train, x_test, params, seed):
        seen.append(len(x_train))
        return np.zeros((len(x_test), 24))

    monkeypatch.setattr(price, "_fit_predict", spy)
    test_days = y.index[-21:]
    rolling_forecast(x, y, test_days, "lear", None, 7, {}, 0)
    # expanding window: training size grows by exactly one block each time
    assert seen == [len(x.loc[x.index < test_days[i]]) for i in (0, 7, 14)]


def test_asinh_scaler_round_trip():
    values = np.random.default_rng(1).normal(40, 30, (200, 3))
    scaler = AsinhScaler().fit(values)
    assert np.allclose(scaler.inverse(scaler.transform(values)), values)


def test_multivariate_dm_detects_better_forecast():
    rng = np.random.default_rng(2)
    truth = rng.normal(0, 1, (300, 24))
    good = truth + rng.normal(0, 0.1, truth.shape)
    bad = truth + rng.normal(0, 1.0, truth.shape)
    assert multivariate_dm(truth, bad, good)[1] < 0.01   # good significantly better
    assert multivariate_dm(truth, good, bad)[1] > 0.99


def test_rolling_fold_keeps_horizon_embargo():
    index = pd.date_range("2024-01-01", periods=24 * 120, freq="h")
    split = make_fold_builder(pd.Timestamp("2024-03-01"), 28, 14)(index, 24)
    assert split.train.max() + pd.Timedelta(hours=24) < split.validation.min()
    assert split.validation.max() + pd.Timedelta(hours=24) < split.test.min()
    assert split.test.min() == pd.Timestamp("2024-03-01")


@pytest.mark.parametrize("model", ["chronos_bolt_small", "chronos2"])
def test_foundation_price_alignment_matches_weekly_lag(fake_pipelines, model):
    hourly = _hourly()
    _, y = build_design(PriceData(hourly, {}, {}), "price_only")
    days = y.index[-10:]
    pred = foundation.predict_price_days(model, hourly, days, {}, False, {"context_hours": 500})
    assert np.allclose(pred, y.shift(7).loc[days].to_numpy())


def test_foundation_price_covariates_respect_declared_lags(fake_pipelines):
    hourly = _hourly()
    days = pd.DatetimeIndex([pd.Timestamp("2024-02-20")])
    foundation.predict_price_days("chronos2", hourly, days, {"Temp": (0,), "Load": (1, 7)}, True,
                                  {"context_hours": 300})
    call = fake_pipelines["chronos2"].calls[0]
    assert set(call["future_covariates"]) == {"Temp"}               # Load is past-only
    assert set(call["past_covariates"]) == {"Temp", "Load"}
    assert len(call["target"]) == 300
    end = hourly.index.get_loc(days[0])
    assert np.allclose(call["future_covariates"]["Temp"], hourly["Temp"].iloc[end : end + 24])
    assert np.allclose(call["target"][-1], hourly["Price"].iloc[end - 1])  # ends at D-1 23:00


@pytest.mark.parametrize("model", ["chronos_bolt_small", "chronos2"])
def test_foundation_load_alignment_matches_seasonal_naive(fake_pipelines, model):
    from open_forecast.models import naive_prediction

    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    rng = np.random.default_rng(3)
    hourly = pd.DataFrame({"Z": rng.normal(100, 10, len(index)),
                           "Z_Dry_Bulb": rng.normal(10, 3, len(index))}, index=index)
    origins = index[400:450]
    pred = foundation.predict_load(model, hourly, origins, ["Z"], ["Z_Dry_Bulb"], 24, {"context_hours": 200})
    expected = naive_prediction("seasonal_naive", hourly, origins, ["Z"], 24)
    assert np.allclose(pred, expected)


def test_epftoolbox_loader_renames_market_headers(tmp_path):
    from open_forecast.price import load_epftoolbox_csv

    index = pd.date_range("2016-01-01", periods=48, freq="h")
    raw = pd.DataFrame({"Zonal COMED price": 1.0, " System load forecast": 2.0,
                        "Zonal COMED load foecast": 3.0}, index=index)
    raw.index.name = "Date"
    raw.to_csv(tmp_path / "PJM.csv")
    data = load_epftoolbox_csv(tmp_path / "PJM.csv")
    assert list(data.hourly.columns) == ["Price", "Exogenous 1", "Exogenous 2"]
    assert data.exogenous_lags == {"Exogenous 1": (0, 1, 7), "Exogenous 2": (0, 1, 7)}
