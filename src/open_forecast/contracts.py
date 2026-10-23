# define the shared data structures
"""Typed records shared across preprocessing, modeling, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetAudit:
    formats_used: dict[str, int]
    first_timestamp: str
    last_timestamp: str
    source_interval: str
    source_rows_per_hour: int
    hourly_rows: int
    renamed_columns: dict[str, str]


@dataclass(frozen=True)
class Split:
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex
    horizon_hours: int
    validation_embargoed_origins: int
    test_embargoed_origins: int

    def manifest(self) -> dict[str, str | int]:
        return {
            "horizon_hours": self.horizon_hours,
            "purge_embargo_hours": self.horizon_hours,
            "train_forecast_origins": len(self.train),
            "validation_forecast_origins": len(self.validation),
            "test_forecast_origins": len(self.test),
            "validation_embargoed_origins": self.validation_embargoed_origins,
            "test_embargoed_origins": self.test_embargoed_origins,
            "train_forecast_origin_end": str(self.train[-1]),
            "validation_forecast_origin_start": str(self.validation[0]),
            "validation_forecast_origin_end": str(self.validation[-1]),
            "test_forecast_origin_start": str(self.test[0]),
            "test_forecast_origin_end": str(self.test[-1]),
            "train_target_end": str(
                self.train[-1] + pd.Timedelta(hours=self.horizon_hours)
            ),
            "validation_target_end": str(
                self.validation[-1] + pd.Timedelta(hours=self.horizon_hours)
            ),
            "test_target_end": str(
                self.test[-1] + pd.Timedelta(hours=self.horizon_hours)
            ),
        }


@dataclass
class TabularPartition:
    x: pd.DataFrame
    y: pd.DataFrame


@dataclass
class SequencePartition:
    x: np.ndarray  # [samples, zones, lookback, features]
    y: np.ndarray  # [samples, zones]
    index: pd.DatetimeIndex
