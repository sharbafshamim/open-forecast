# set up the tree models and baselines
"""Tabular model adapters and transparent naive baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.multioutput import MultiOutputRegressor

from ..contracts import TabularPartition


class TreeForecaster:
    """Standardizes the tree-model adapter contract."""

    def __init__(self, name: str, parameters: dict, seed: int) -> None:
        self.name = name

        if name == "random_forest":
            base = RandomForestRegressor(
                **parameters,
                n_jobs=-1,
                random_state=seed,
            )
        elif name == "extra_trees":
            base = ExtraTreesRegressor(
                **parameters,
                n_jobs=-1,
                random_state=seed,
            )
        elif name == "hist_gradient_boosting":
            base = HistGradientBoostingRegressor(
                **parameters,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown tree model: {name}")

        self.model = MultiOutputRegressor(base)

    def fit(self, partition: TabularPartition) -> "TreeForecaster":
        self.model.fit(partition.x, partition.y)
        return self

    def predict(self, partition: TabularPartition) -> np.ndarray:
        return self.model.predict(partition.x)

    def fit_predict_lean(
        self, train: TabularPartition, partitions: list[TabularPartition]
    ) -> list[np.ndarray]:
        """Memory-lean equivalent of fit() followed by predict() on each partition.

        Fits one clone of the base estimator per target zone (exactly as
        MultiOutputRegressor does, same parameters and random_state), predicts,
        and discards the fitted clone before the next zone. Predictions are
        identical; peak memory holds one zone's model instead of all zones'.
        """
        from sklearn.base import clone
        outputs = [np.empty((len(p.x), train.y.shape[1])) for p in partitions]
        for column in range(train.y.shape[1]):
            estimator = clone(self.model.estimator)
            estimator.fit(train.x, train.y.iloc[:, column])
            for out, part in zip(outputs, partitions):
                out[:, column] = estimator.predict(part.x)
            del estimator
        return outputs


def naive_prediction(
    name: str,
    hourly: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    targets: list[str],
    horizon_hours: int,
) -> np.ndarray:
    """
    Return persistence or daily seasonal-naive predictions at the same origin.

    Because target_index stores forecast origins, t is a persistence prediction.
    At a 24-hour horizon a daily seasonal-naive baseline would be identical to
    persistence, so the seasonal baseline uses the corresponding target time
    one week earlier: (t + 24h) - 168h = t - 144h.
    """
    if name == "persistence":
        offset = 0
    elif name == "seasonal_naive":
        offset = 168 - horizon_hours
    else:
        raise ValueError(f"Unknown naive model: {name}")

    source_index = target_index - pd.Timedelta(hours=offset)
    return hourly.loc[source_index, targets].to_numpy()
