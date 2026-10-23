"""Zero-shot time-series foundation models under the framework's information sets.

Supported models (weights download from the Hugging Face Hub on first use):

``chronos_bolt_small`` / ``chronos_bolt_base``
    amazon/chronos-bolt-{small,base}. Univariate: sees only the target history.
``chronos2``
    amazon/chronos-2. Accepts covariates: past-only covariates and covariates
    known over the forecast horizon. Used here with exactly the exogenous
    inputs the feature-based models receive, so the comparison isolates the
    learner rather than the information set.

No model is fine-tuned: every forecast is zero-shot from a context window that
ends at the forecast origin (load) or at the end of day D-1 (price).

``pipeline_factory`` can be overridden (tests inject a deterministic fake that
reproduces the seasonal-naive forecast, proving index alignment end to end).
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd

FOUNDATION_MODELS = {
    "chronos_bolt_small": "amazon/chronos-bolt-small",
    "chronos_bolt_base": "amazon/chronos-bolt-base",
    "chronos2": "amazon/chronos-2",
}
COVARIATE_MODELS = {"chronos2"}
_PIPELINES: dict[str, object] = {}


def _default_factory(model: str, device: str):
    import torch
    from chronos import BaseChronosPipeline

    dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
    return BaseChronosPipeline.from_pretrained(FOUNDATION_MODELS[model], device_map=device, torch_dtype=dtype)


pipeline_factory: Callable[[str, str], object] = _default_factory


def get_pipeline(model: str, device: str = "cpu"):
    if model not in FOUNDATION_MODELS:
        raise ValueError(f"Unknown foundation model: {model}")
    key = f"{model}@{device}"
    if key not in _PIPELINES:
        _PIPELINES[key] = pipeline_factory(model, device)
    return _PIPELINES[key]


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=float)


def forecast_batch(
    model: str,
    contexts: list[np.ndarray],
    horizon: int,
    past_covariates: list[dict[str, np.ndarray]] | None = None,
    future_covariates: list[dict[str, np.ndarray]] | None = None,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Median forecasts, shape [len(contexts), horizon]."""
    pipeline = get_pipeline(model, device)
    outputs = []
    for start in range(0, len(contexts), batch_size):
        stop = start + batch_size
        if model in COVARIATE_MODELS:
            inputs = []
            for i in range(start, min(stop, len(contexts))):
                item = {"target": contexts[i].astype(np.float32)}
                if past_covariates is not None:
                    item["past_covariates"] = {k: v.astype(np.float32) for k, v in past_covariates[i].items()}
                    item["future_covariates"] = {k: v.astype(np.float32) for k, v in future_covariates[i].items()}
                inputs.append(item)
            quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=[0.5])
            # list of [n_variates=1, horizon, 1]
            outputs.extend(_to_numpy(q)[0, :, 0] for q in quantiles)
        else:
            try:
                import torch

                batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[start:stop]]
            except ImportError:  # only reachable with an injected test pipeline
                batch = [c.astype(np.float32) for c in contexts[start:stop]]
            quantiles, _ = pipeline.predict_quantiles(batch, prediction_length=horizon, quantile_levels=[0.5])
            outputs.extend(_to_numpy(quantiles)[:, :, 0])
    return np.vstack(outputs)


# --------------------------------------------------------------------------
# Load harness: direct t+H point forecast at every hourly forecast origin
# --------------------------------------------------------------------------
def predict_load(
    model: str,
    hourly: pd.DataFrame,
    origins: pd.DatetimeIndex,
    targets: list[str],
    weather: list[str],
    horizon: int,
    settings: dict,
) -> np.ndarray:
    """Return y_hat(t + H) for every origin t and zone, shape [origins, zones].

    Context = the last ``context_hours`` observations ending at t (inclusive).
    chronos2 additionally receives the zone's weather columns as covariates:
    past values over the context and the declared target-time weather proxy
    over t+1..t+H (the same proxy the feature-based models use).
    """
    context = settings.get("context_hours", 512)
    device = settings.get("device", os.environ.get("OPEN_FORECAST_DEVICE", "cpu"))
    batch_size = settings.get("batch_size", 256)
    positions = hourly.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("Forecast origin missing from the hourly index.")
    result = np.zeros((len(origins), len(targets)))
    for z, zone in enumerate(targets):
        series = hourly[zone].to_numpy(dtype=float)
        zone_weather = [c for c in weather if c.startswith(f"{zone}_")] or list(weather)
        contexts, past, future = [], [], []
        for p in positions:
            lo = max(0, p - context + 1)
            contexts.append(series[lo : p + 1])
            if model in COVARIATE_MODELS:
                past.append({c: hourly[c].to_numpy()[lo : p + 1] for c in zone_weather})
                future.append({c: hourly[c].to_numpy()[p + 1 : p + 1 + horizon] for c in zone_weather})
        use_cov = model in COVARIATE_MODELS
        path = forecast_batch(model, contexts, horizon, past if use_cov else None,
                              future if use_cov else None, device, batch_size)
        result[:, z] = path[:, horizon - 1]
    return result


# --------------------------------------------------------------------------
# Price harness: 24 hourly prices of day D from information up to end of D-1
# --------------------------------------------------------------------------
def predict_price_days(
    model: str,
    hourly: pd.DataFrame,
    days: pd.DatetimeIndex,
    exogenous_lags: dict[str, tuple[int, ...]],
    use_exogenous: bool,
    settings: dict,
) -> np.ndarray:
    """Return [len(days), 24] forecasts.

    With ``use_exogenous`` (chronos2 only), every exogenous column is a past
    covariate; columns whose declared lags include 0 (known for day D) are also
    supplied as future covariates. Past-only columns (e.g. ISO-NE RT_Demand)
    are never given for day D.
    """
    context = settings.get("context_hours", 2048)
    device = settings.get("device", os.environ.get("OPEN_FORECAST_DEVICE", "cpu"))
    price = hourly["Price"].to_numpy(dtype=float)
    contexts, past, future = [], [], []
    for day in days:
        end = hourly.index.get_indexer([day])[0]          # first hour of day D
        if end < 0:
            raise ValueError(f"Day {day} not in hourly index")
        lo = max(0, end - context)
        contexts.append(price[lo:end])
        if use_exogenous:
            past.append({c: hourly[c].to_numpy()[lo:end] for c in exogenous_lags})
            future.append({c: hourly[c].to_numpy()[end : end + 24]
                           for c, lags in exogenous_lags.items() if 0 in lags})
    if use_exogenous and model not in COVARIATE_MODELS:
        raise ValueError(f"{model} cannot use exogenous covariates")
    return forecast_batch(model, contexts, 24, past if use_exogenous else None,
                          future if use_exogenous else None, device, settings.get("batch_size", 64))
