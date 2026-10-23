# load and clean the raw timestamps
"""Dataset loading with explicit source-time semantics.

Fix: the public Tetouan City release ships target columns as \"Zone 1 Power
Consumption\", \"Zone 2  Power Consumption\" (note the double space) and
\"Zone 3  Power Consumption\", plus a derived \"Sum\" column. This loader
normalizes column names on load (collapses internal whitespace, strips a
trailing \"Power Consumption\" suffix) so the raw, unmodified CSV from the
source works with plain \"Zone 1\"/\"Zone 2\" names in the config, without any
manual renaming step.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .contracts import DatasetAudit


HYPHEN_FORMAT = "%m-%d-%Y %H:%M"
SLASH_FORMAT = "%m/%d/%Y %H:%M"

_WHITESPACE = re.compile(r"\s+")
_POWER_CONSUMPTION_SUFFIX = re.compile(r"\s+Power Consumption$", re.IGNORECASE)


def _normalize_column_name(name: str) -> str:
    """Collapse whitespace and drop a trailing '... Power Consumption' suffix.

    "Zone 2  Power Consumption" -> "Zone 2"
    "general diffuse flows"     -> "general diffuse flows" (unchanged)
    """
    collapsed = _WHITESPACE.sub(" ", str(name).strip())
    return _POWER_CONSUMPTION_SUFFIX.sub("", collapsed).strip()


def _normalize_columns(columns: pd.Index) -> tuple[pd.Index, dict[str, str]]:
    renamed = {}
    normalized = []
    for original in columns:
        fixed = _normalize_column_name(original)
        normalized.append(fixed)
        if fixed != original:
            renamed[str(original)] = fixed
    return pd.Index(normalized), renamed


def _parse_datetime(values: pd.Series) -> tuple[pd.DatetimeIndex, dict[str, int]]:
    """Parse the Tetouan source's documented month-day-year representations."""
    text = values.astype(str).str.strip()
    hyphen = text.str.contains("-", regex=False)
    slash = text.str.contains("/", regex=False)

    if (~(hyphen | slash)).any():
        examples = text.loc[~(hyphen | slash)].head(3).tolist()
        raise ValueError(f"Unsupported datetime representation: {examples}")

    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    parsed.loc[hyphen] = pd.to_datetime(
        text.loc[hyphen], format=HYPHEN_FORMAT, errors="raise"
    )
    parsed.loc[slash] = pd.to_datetime(
        text.loc[slash], format=SLASH_FORMAT, errors="raise"
    )
    index = pd.DatetimeIndex(parsed)

    if not index.is_monotonic_increasing:
        raise ValueError(
            "Datetime policy produced non-monotonic rows. Refusing to sort because "
            "sorting would hide a source-format or ingestion error."
        )

    if index.has_duplicates:
        raise ValueError("Source contains duplicate timestamps.")

    return index, {
        "month-day-year (hyphen)": int(hyphen.sum()),
        "month/day/year (slash)": int(slash.sum()),
    }


def load_hourly_dataset(
    path: str | Path,
    datetime_column: str,
    target_columns: list[str],
    weather_columns: list[str],
) -> tuple[pd.DataFrame, DatasetAudit]:
    """Read, validate, normalize, and hourly-resample the source CSV."""
    raw = pd.read_csv(path)
    raw.columns, renamed_columns = _normalize_columns(raw.columns)

    required = {datetime_column, *target_columns, *weather_columns}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"Dataset lacks required columns: {sorted(missing)}. "
            f"Columns available after normalization: {sorted(raw.columns)}"
        )

    index, formats = _parse_datetime(raw[datetime_column])
    intervals = pd.Series(index[1:] - index[:-1])
    source_interval = intervals.mode().iloc[0]

    if not (intervals == source_interval).all():
        raise ValueError(
            "Source cadence has gaps or irregular intervals. Refusing to "
            "resample because shifted targets would no longer be elapsed hours."
        )

    if pd.Timedelta(hours=1) % source_interval != pd.Timedelta(0):
        raise ValueError(f"Source interval {source_interval} does not divide one hour.")

    source_rows_per_hour = int(pd.Timedelta(hours=1) / source_interval)

    data = raw.drop(columns=datetime_column)
    data.index = index

    required_numeric = [*target_columns, *weather_columns]
    hourly_counts = data[required_numeric].resample("h").count()

    if not (hourly_counts == source_rows_per_hour).all().all():
        raise ValueError(
            "At least one required column has an incomplete hour. Refusing to "
            "average partial hours without an explicit imputation policy."
        )

    hourly = data[required_numeric].resample("h").mean()

    if hourly.isna().any().any():
        raise ValueError("Hourly data contains missing values after validated resampling.")

    expected_hourly = pd.date_range(
        hourly.index.min(), hourly.index.max(), freq="h"
    )
    if not hourly.index.equals(expected_hourly):
        raise ValueError("Hourly index is not continuous.")

    if len(hourly) < 24 * 30:
        raise ValueError("Fewer than 30 days of complete hourly values remain.")

    audit = DatasetAudit(
        formats_used=formats,
        first_timestamp=str(index.min()),
        last_timestamp=str(index.max()),
        source_interval=str(source_interval),
        source_rows_per_hour=source_rows_per_hour,
        hourly_rows=len(hourly),
        renamed_columns=renamed_columns,
    )
    return hourly, audit
