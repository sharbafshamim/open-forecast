# build the feature sets and splits
"""Feature-availability scenarios and common chronological data partitions."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .contracts import SequencePartition, Split, TabularPartition


FeatureMode = Literal["weather_time", "weather_time_history", "weather_time_history_weekly"]

HISTORY_MODES = {"weather_time_history", "weather_time_history_weekly"}
# Mode B+ adds the target-aligned weekly lags. At a 24-h horizon the lag of
# 144 h equals y(t + H - 168 h): the value the seasonal-naive benchmark uses.
DEFAULT_WEEKLY_LAGS = (144, 168)


def chronological_split(
    index: pd.DatetimeIndex,
    horizon_hours: int,
    train_fraction: float,
    validation_fraction: float,
) -> Split:
    """Split forecast origins without randomization."""
    if not index.is_monotonic_increasing:
        raise ValueError("Forecast origin index must be ordered.")

    if len(index) < 3 * horizon_hours + 10:
        raise ValueError(
            "Too few forecast origins to build horizon-purged partitions."
        )

    train_end = int(len(index) * train_fraction)
    validation_end = train_end + int(len(index) * validation_fraction)
    validation_start = train_end + horizon_hours
    test_start = validation_end + horizon_hours

    if validation_start >= validation_end or test_start >= len(index):
        raise ValueError(
            "Split fractions leave no observations after the horizon embargo."
        )

    return Split(
        train=index[:train_end],
        validation=index[validation_start:validation_end],
        test=index[test_start:],
        horizon_hours=horizon_hours,
        validation_embargoed_origins=horizon_hours,
        test_embargoed_origins=horizon_hours,
    )


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * index.hour / 24),
            "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
            "is_weekend": (index.dayofweek >= 5).astype(float),
        },
        index=index,
    )


def build_tabular_frame(
    hourly: pd.DataFrame,
    targets: list[str],
    weather: list[str],
    mode: FeatureMode,
    horizon_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    weekly_lags: list[int] | tuple[int, ...] = DEFAULT_WEEKLY_LAGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a direct t+H tabular task.

    Mode A uses target-time calendar features and observed target-time weather
    as the original study's documented weather-forecast proxy. Mode B adds
    origin-time load lags/rolling values. The proxy is a declared experimental
    assumption, never a silent production claim.
    """
    target_time = hourly.index + pd.Timedelta(hours=horizon_hours)
    future_weather = hourly[weather].shift(-horizon_hours).add_prefix("target_weather_")
    calendar = _calendar_features(target_time)
    calendar.index = hourly.index

    x = pd.concat([future_weather, calendar], axis=1)

    if mode in HISTORY_MODES:
        for zone in targets:
            for lag in lags:
                x[f"{zone}_lag_{lag}h"] = hourly[zone].shift(lag)
            for window in rolling_windows:
                x[f"{zone}_rolling_mean_{window}h"] = (
                    hourly[zone].rolling(window).mean()
                )
            if mode == "weather_time_history_weekly":
                for lag in weekly_lags:
                    x[f"{zone}_weekly_lag_{lag}h"] = hourly[zone].shift(lag)
    elif mode != "weather_time":
        raise ValueError(f"Unknown feature mode: {mode}")

    y = hourly[targets].shift(-horizon_hours)
    usable = x.notna().all(axis=1) & y.notna().all(axis=1)
    return x.loc[usable], y.loc[usable]


def tabular_partitions(
    x: pd.DataFrame,
    y: pd.DataFrame,
    split: Split,
) -> dict[str, TabularPartition]:
    """Return fit partitions keyed by train/validation/test."""
    return {
        "train": TabularPartition(x.loc[split.train], y.loc[split.train]),
        "validation": TabularPartition(
            x.loc[split.validation], y.loc[split.validation]
        ),
        "test": TabularPartition(x.loc[split.test], y.loc[split.test]),
    }


def build_sequence_partitions(
    hourly: pd.DataFrame,
    targets: list[str],
    weather: list[str],
    mode: FeatureMode,
    horizon_hours: int,
    lookback_hours: int,
    split: Split,
    weekly_lags: list[int] | tuple[int, ...] = DEFAULT_WEEKLY_LAGS,
) -> dict[str, SequencePartition]:
    """
    Create [sample, zone, lookback, feature] tensors for recurrent/Transformer models.

    Exogenous inputs are shared across zones. In Mode B each zone additionally
    receives its own consumption history, preserving the controlled
    feature-availability question without leaking the future target.
    """
    target_time = hourly.index + pd.Timedelta(hours=horizon_hours)
    exogenous = pd.concat(
        [
            # Same declared target-time weather proxy used by tabular models.
            # At a final sequence step t it provides the proxy for target t+H.
            hourly[weather].shift(-horizon_hours),
            _calendar_features(target_time).set_index(hourly.index),
        ],
        axis=1,
    )

    exo_values = exogenous.to_numpy(dtype=np.float32)
    target_values = hourly[targets].to_numpy(dtype=np.float32)
    # Weekly-lag channels for Mode B+: at sequence step s the channel holds
    # y(s - lag). Early rows are NaN and are excluded by the common origin set.
    weekly_values = [
        hourly[targets].shift(lag).to_numpy(dtype=np.float32) for lag in weekly_lags
    ]

    origins: list[pd.Timestamp] = []
    tensors: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for end in range(lookback_hours - 1, len(hourly) - horizon_hours):
        sequence_start = end - lookback_hours + 1
        per_zone = []

        for zone_index in range(len(targets)):
            zone_exogenous = exo_values[sequence_start : end + 1]
            if mode in HISTORY_MODES:
                history = target_values[
                    sequence_start : end + 1, zone_index : zone_index + 1
                ]
                zone_exogenous = np.concatenate(
                    [zone_exogenous, history], axis=1
                )
                if mode == "weather_time_history_weekly":
                    weekly = [
                        values[sequence_start : end + 1, zone_index : zone_index + 1]
                        for values in weekly_values
                    ]
                    zone_exogenous = np.concatenate([zone_exogenous, *weekly], axis=1)
            elif mode != "weather_time":
                raise ValueError(f"Unknown feature mode: {mode}")

            per_zone.append(zone_exogenous)

        origins.append(hourly.index[end])
        tensors.append(np.stack(per_zone))
        labels.append(target_values[end + horizon_hours])

    x = np.stack(tensors).astype(np.float32)
    y = np.stack(labels).astype(np.float32)
    index = pd.DatetimeIndex(origins)

    partitions: dict[str, SequencePartition] = {}
    for name, partition_index in {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }.items():
        mask = index.isin(partition_index)
        partitions[name] = SequencePartition(
            x=x[mask], y=y[mask], index=index[mask]
        )

    return partitions


class SequenceScaler:
    """Training-only standardization for sequence tensors and multi-zone targets."""

    def __init__(self) -> None:
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit_transform(
        self, partition: SequencePartition
    ) -> SequencePartition:
        self.x_scaler.fit(
            partition.x.reshape(-1, partition.x.shape[-1])
        )
        self.y_scaler.fit(partition.y)
        return self.transform(partition)

    def transform(self, partition: SequencePartition) -> SequencePartition:
        scaled_x = self.x_scaler.transform(
            partition.x.reshape(-1, partition.x.shape[-1])
        ).reshape(partition.x.shape)

        return SequencePartition(
            x=scaled_x.astype(np.float32),
            y=self.y_scaler.transform(partition.y).astype(np.float32),
            index=partition.index,
        )

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(values)
