
import numpy as np
import pandas as pd
import pytest

from open_forecast.evaluation import metric_rows
from open_forecast.features import (
    build_sequence_partitions,
    chronological_split,
)
from open_forecast.preprocessing import (
    _parse_datetime,
    load_hourly_dataset,
)


# 1. source parser
def test_parser_accepts_mixed_timestamp_formats():
    values = pd.Series(
        [
            "01-12-2017 23:50",
            "1/13/2017 0:00",
            "1/13/2017 0:10",
        ]
    )

    index, formats = _parse_datetime(values)

    assert index.is_monotonic_increasing
    assert not index.has_duplicates
    assert formats["month-day-year (hyphen)"] == 1
    assert formats["month/day/year (slash)"] == 2
    assert index[0] == pd.Timestamp("2017-01-12 23:50")
    assert index[1] == pd.Timestamp("2017-01-13 00:00")


# 2. non-monotonic input refusal
def test_parser_rejects_non_monotonic_input():
    values = pd.Series(
        [
            "01-13-2017 00:10",
            "01-13-2017 00:00",
            "01-13-2017 00:20",
        ]
    )

    with pytest.raises(ValueError, match="non-monotonic"):
        _parse_datetime(values)


# 3. incomplete required-column detection
def test_loader_rejects_incomplete_required_hour(tmp_path):
    periods = 31 * 24 * 6

    timestamps = pd.date_range(
        "2017-01-01 00:00",
        periods=periods,
        freq="10min",
    )

    frame = pd.DataFrame(
        {
            "DateTime": timestamps.strftime("%m-%d-%Y %H:%M"),
            "Zone 1 Power Consumption": np.arange(periods, dtype=float),
            "Zone 2  Power Consumption": np.arange(periods, dtype=float) + 10,
            "Temperature": np.full(periods, 20.0),
            "Humidity": np.full(periods, 50.0),
            "Wind Speed": np.full(periods, 2.0),
            "general diffuse flows": np.full(periods, 100.0),
            "diffuse flows": np.full(periods, 50.0),
        }
    )

    # One required observation is missing, so this hour must be rejected.
    frame.loc[2, "Temperature"] = np.nan

    path = tmp_path / "incomplete.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="incomplete hour"):
        load_hourly_dataset(
            path,
            datetime_column="DateTime",
            target_columns=["Zone 1", "Zone 2"],
            weather_columns=[
                "Temperature",
                "Humidity",
                "Wind Speed",
                "general diffuse flows",
                "diffuse flows",
            ],
        )


# 4. horizon embargoes
def test_chronological_split_applies_horizon_embargo():
    index = pd.date_range(
        "2017-01-01",
        periods=1000,
        freq="h",
    )

    horizon = 24

    split = chronological_split(
        index=index,
        horizon_hours=horizon,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    train_end = int(len(index) * 0.70)
    validation_end = train_end + int(len(index) * 0.15)

    assert split.validation[0] == index[train_end + horizon]
    assert split.test[0] == index[validation_end + horizon]
    assert split.validation_embargoed_origins == horizon
    assert split.test_embargoed_origins == horizon

    assert (
        split.train[-1] + pd.Timedelta(hours=horizon)
        < split.validation[0]
    )

    assert (
        split.validation[-1] + pd.Timedelta(hours=horizon)
        < split.test[0]
    )


# 5. target-time weather alignment for sequences
def test_sequence_target_weather_alignment():
    index = pd.date_range(
        "2017-01-01",
        periods=240,
        freq="h",
    )

    hourly = pd.DataFrame(
        {
            "Zone 1": np.arange(240, dtype=float),
            "Zone 2": np.arange(240, dtype=float) + 1000,
            "Temperature": np.arange(240, dtype=float) + 5000,
            "Humidity": np.arange(240, dtype=float) + 6000,
        },
        index=index,
    )

    split = chronological_split(
        index=index,
        horizon_hours=24,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    partitions = build_sequence_partitions(
        hourly=hourly,
        targets=["Zone 1", "Zone 2"],
        weather=["Temperature", "Humidity"],
        mode="weather_time",
        horizon_hours=24,
        lookback_hours=24,
        split=split,
    )

    train = partitions["train"]
    assert len(train.index) > 0

    first_origin = train.index[0]

    # Feature 0 is Temperature. At the final sequence step for origin t,
    # it must contain weather from target time t+24.
    observed_feature = train.x[0, 0, -1, 0]

    expected_weather = hourly.loc[
        first_origin + pd.Timedelta(hours=24),
        "Temperature",
    ]

    assert np.isclose(observed_feature, expected_weather)


# 6. preservation of per-zone metric rows
def test_metric_rows_preserve_each_zone():
    actual = np.array(
        [
            [10.0, 20.0],
            [12.0, 18.0],
            [15.0, 25.0],
            [17.0, 22.0],
        ]
    )

    prediction = np.array(
        [
            [11.0, 19.0],
            [11.0, 20.0],
            [14.0, 24.0],
            [18.0, 23.0],
        ]
    )

    metrics = metric_rows(
        model="test_model",
        mode="weather_time",
        split="test",
        actual=actual,
        prediction=prediction,
        zones=["Zone 1", "Zone 2"],
    )

    assert set(metrics["scope"]) == {
        "Zone 1",
        "Zone 2",
        "aggregate",
    }

    assert (metrics["scope"] == "Zone 1").sum() == 1
    assert (metrics["scope"] == "Zone 2").sum() == 1
    assert (metrics["scope"] == "aggregate").sum() == 1

    assert {
        "rmse",
        "mae",
        "nrmse_pct",
        "r2",
    }.issubset(metrics.columns)
