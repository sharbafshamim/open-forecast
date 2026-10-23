# calculate the evaluation metrics
"""Standardized metrics and result artifacts for every model/scenario.

Metrics are reported in the target's own unit (declared per dataset as
``dataset.unit`` in the config: MW for ISO-NE; the Tetouan source states no
unit) plus ``nrmse_pct`` = 100 * RMSE / mean(|actual|), which is unit-free and
comparable across datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metric_rows(
    model: str,
    mode: str,
    split: str,
    actual: np.ndarray,
    prediction: np.ndarray,
    zones: list[str],
) -> pd.DataFrame:
    """Compute mandatory per-zone rows plus an explicit aggregate diagnostic."""
    rows: list[dict[str, float | str]] = []

    for position, zone in enumerate(zones):
        y_true = actual[:, position]
        y_pred = prediction[:, position]
        rows.append(
            {
                "model": model,
                "feature_mode": mode,
                "split": split,
                "scope": zone,
                "rmse": float(
                    np.sqrt(mean_squared_error(y_true, y_pred))
                ),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "nrmse_pct": float(
                    100 * np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(np.abs(y_true))
                ),
                "r2": float(r2_score(y_true, y_pred)),
            }
        )

    rows.append(
        {
            "model": model,
            "feature_mode": mode,
            "split": split,
            "scope": "aggregate",
            "rmse": float(
                np.sqrt(mean_squared_error(actual, prediction))
            ),
            "mae": float(
                mean_absolute_error(actual, prediction)
            ),
            "nrmse_pct": float(
                100 * np.sqrt(mean_squared_error(actual, prediction)) / np.mean(np.abs(actual))
            ),
            "r2": float(np.mean([row["r2"] for row in rows])),
        }
    )

    return pd.DataFrame(rows)


def prediction_rows(
    model: str,
    mode: str,
    split: str,
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    prediction: np.ndarray,
    zones: list[str],
    horizon_hours: int,
) -> pd.DataFrame:
    """Return long-form target records so every plotted point can be inspected."""
    offset = pd.Timedelta(hours=horizon_hours)
    rows = []

    for row_index, forecast_origin in enumerate(index):
        for zone_index, zone in enumerate(zones):
            rows.append(
                {
                    "model": model,
                    "feature_mode": mode,
                    "split": split,
                    "forecast_origin": forecast_origin,
                    "target_time": forecast_origin + offset,
                    "zone": zone,
                    "actual": actual[row_index, zone_index],
                    "predicted": prediction[row_index, zone_index],
                }
            )

    return pd.DataFrame(rows)
