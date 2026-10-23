"""ISO New England SMD Hourly loader with an explicit datetime policy.

Source: ISO-NE "SMD Hourly Data" annual workbooks (one sheet per load zone).

Declared policy
---------------
* Hr_End is hour-ending on the US/Eastern local clock.
* Files from 2024 onward follow the true clock: a 23-hour day in March and a
  25-hour day in November, whose repeated hour is labelled "02X".
* Earlier files (here 2023) use a legacy convention: every day has 24 rows.
  In March, HE02 is a synthetic average of HE01 and HE03; it is dropped. In
  November, the two repeated hours are averaged into one HE02 row; that row is
  kept as the first occurrence and the second occurrence is linearly
  interpolated. Every interpolated hour is counted in the audit.
* Timestamps are hour-beginning and expressed on a fixed UTC-05:00 clock
  (Eastern Standard Time, no daylight saving) so the hourly index is
  continuous. Calendar features are therefore computed on standard time.
* Only RT_Demand (target, MW), Dry_Bulb and Dew_Point (degF) are used.
  DA_Demand (market-cleared day-ahead quantity including virtual bids) and all
  price columns are excluded because they would change the information set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import DatasetAudit

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA"]
WEATHER = ["Dry_Bulb", "Dew_Point"]
LOCAL_TZ = "America/New_York"
FIXED_OFFSET = pd.Timedelta(hours=-5)


def _one_sheet(path: Path, zone: str) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_excel(path, sheet_name=zone, usecols=["Date", "Hr_End", "RT_Demand", *WEATHER])
    frame["h"] = frame["Hr_End"].astype(str).str.strip()
    frame["Date"] = pd.to_datetime(frame["Date"])
    counts = {"true_clock_days": 0, "legacy_spring_rows_dropped": 0, "legacy_fall_hours_interpolated": 0}
    rows = []
    for day, group in frame.groupby("Date", sort=True):
        group = group.copy()
        start = pd.Timestamp(day).tz_localize(LOCAL_TZ)
        end = pd.Timestamp(day + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
        real_hours = int((end - start) / pd.Timedelta(hours=1))
        if real_hours != 24 and len(group) == real_hours:
            counts["true_clock_days"] += 1
        if len(group) == 24 and real_hours == 23:
            group = group[group["h"].str.lstrip("0") != "2"]
            counts["legacy_spring_rows_dropped"] += 1
        if len(group) == 24 and real_hours == 25:
            position = group.index[group["h"].str.lstrip("0") == "2"][0]
            gap = group.loc[[position]].copy()
            gap[["RT_Demand", *WEATHER]] = np.nan
            group = pd.concat([group.loc[:position], gap, group.loc[position + 1 :]])
            counts["legacy_fall_hours_interpolated"] += 1
        if len(group) != real_hours:
            raise ValueError(f"{path.name} {zone} {day.date()}: {len(group)} rows for {real_hours} real hours")
        utc = start.tz_convert("UTC") + pd.to_timedelta(np.arange(real_hours), unit="h")
        group.index = (utc + FIXED_OFFSET).tz_localize(None)
        rows.append(group)
    data = pd.concat(rows)[["RT_Demand", *WEATHER]]
    return data, counts


def load_isone_smd(paths: list[str | Path], zones: list[str] | None = None) -> tuple[pd.DataFrame, DatasetAudit]:
    zones = zones or ZONES
    columns = {}
    totals: dict[str, int] = {}
    for zone in zones:
        parts = []
        for path in paths:
            data, counts = _one_sheet(Path(path), zone)
            parts.append(data)
            for key, value in counts.items():
                totals[f"{zone}:{key}"] = totals.get(f"{zone}:{key}", 0) + value
        zone_frame = pd.concat(parts)
        if zone_frame.index.has_duplicates:
            raise ValueError(f"{zone}: duplicate hours after DST normalisation")
        columns[zone] = zone_frame["RT_Demand"]
        for variable in WEATHER:
            columns[f"{zone}_{variable}"] = zone_frame[variable]
    hourly = pd.DataFrame(columns)
    expected = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(expected)
    missing_before = int(hourly.isna().sum().sum())
    hourly = hourly.interpolate(limit=2)
    if hourly.isna().any().any():
        raise ValueError("Missing values remain after the declared interpolation policy.")
    totals["cells_interpolated"] = missing_before
    audit = DatasetAudit(
        formats_used=totals,
        first_timestamp=str(hourly.index.min()),
        last_timestamp=str(hourly.index.max()),
        source_interval="1:00:00",
        source_rows_per_hour=1,
        hourly_rows=len(hourly),
        renamed_columns={"RT_Demand": "<zone>", "Dry_Bulb": "<zone>_Dry_Bulb", "Dew_Point": "<zone>_Dew_Point"},
    )
    return hourly, audit
