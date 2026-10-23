"""Tests for Mode B+ (weekly lags) and the ISO-NE daylight-saving policy."""

import numpy as np
import pandas as pd

from open_forecast.features import build_tabular_frame


def test_weekly_lag_equals_seasonal_naive_input():
    index = pd.date_range("2024-01-01", periods=24 * 20, freq="h")
    hourly = pd.DataFrame({"Z": np.arange(len(index), dtype=float), "T": 1.0}, index=index)
    x, y = build_tabular_frame(hourly, ["Z"], ["T"], "weather_time_history_weekly", 24, [1], [3], [144, 168])
    origin = x.index[0]
    # lag 144 at origin t is y(t + 24h - 168h), the seasonal-naive input
    assert x.loc[origin, "Z_weekly_lag_144h"] == hourly.loc[origin + pd.Timedelta(hours=24 - 168), "Z"]
    assert y.loc[origin, "Z"] == hourly.loc[origin + pd.Timedelta(hours=24), "Z"]
    # Mode B frames never contain weekly columns
    xb, _ = build_tabular_frame(hourly, ["Z"], ["T"], "weather_time_history", 24, [1], [3])
    assert not any("weekly" in c for c in xb.columns)


def _write_smd(path, rows):
    frame = pd.DataFrame(rows, columns=["Date", "Hr_End", "RT_Demand", "Dry_Bulb", "Dew_Point"])
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="ME", index=False)


def test_isone_true_clock_and_legacy_dst_give_continuous_hours(tmp_path):
    from open_forecast.isone import load_isone_smd

    # legacy 2023 convention: 24 rows on the spring-forward day (HE02 synthetic)
    legacy = [("2023-03-12", h, 100.0 + h, 30, 20) for h in range(1, 25)]
    legacy += [("2023-03-13", h, 200.0 + h, 30, 20) for h in range(1, 25)]
    # true-clock 2024 convention: 25 rows with "02X" on the fall-back day
    labels = ["01", "02", "02X"] + [f"{h:02d}" for h in range(3, 25)]
    true_clock = [("2024-11-03", lab, 300.0 + i, 40, 30) for i, lab in enumerate(labels)]
    _write_smd(tmp_path / "a.xlsx", legacy)
    _write_smd(tmp_path / "b.xlsx", true_clock)

    hourly_a, audit_a = load_isone_smd([tmp_path / "a.xlsx"], ["ME"])
    assert len(hourly_a) == 23 + 24                      # synthetic hour dropped
    assert 102.0 not in hourly_a["ME"].values            # the HE02 average is gone
    assert audit_a.formats_used["ME:legacy_spring_rows_dropped"] == 1

    hourly_b, _ = load_isone_smd([tmp_path / "b.xlsx"], ["ME"])
    assert len(hourly_b) == 25                           # both repeated hours kept
    assert hourly_b.index.is_monotonic_increasing and not hourly_b.index.has_duplicates
    assert hourly_b["ME"].tolist() == [300.0 + i for i in range(25)]
