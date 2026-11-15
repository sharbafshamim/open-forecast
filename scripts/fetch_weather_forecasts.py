"""Download archived day-ahead weather FORECASTS for the ISO-NE load zones.

Source: Open-Meteo Previous Runs API (NCEP GFS). ``<variable>_previous_day2``
is the value forecast by a model run issued at least 48 h before the valid
time, so it was available at every forecast origin t <= valid time - 24 h.
That makes it a strictly valid day-ahead weather input for t+1 .. t+24
(``--lead 1`` gives the shorter-lead, slightly optimistic variant).

Output: data/raw/isone/weather_forecasts_previous_day{lead}.csv with a ``ts``
column on the fixed UTC-05:00 clock used by the ISO-NE loader and one column
per zone variable, in degrees Fahrenheit like the SMD files.

Usage:  python scripts/fetch_weather_forecasts.py [--lead 2] [--start 2024-12-01] [--end 2025-12-31]

Requests are made one month at a time with retries, because year-long requests time out.

The station of each zone is the zone's main airport (listed below); the
script prints the correlation of each forecast series with the observed SMD
weather so the mapping can be checked.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ZONE_STATIONS = {               # (latitude, longitude) of a representative airport
    "ME": (43.646, -70.309),    # Portland
    "NH": (43.195, -71.501),    # Concord
    "VT": (44.468, -73.150),    # Burlington
    "CT": (41.939, -72.683),    # Windsor Locks (Bradley)
    "RI": (41.724, -71.428),    # Providence (T.F. Green)
    "SEMA": (41.676, -70.958),  # New Bedford
    "WCMA": (42.267, -71.876),  # Worcester
    "NEMA": (42.364, -71.005),  # Boston (Logan)
}
VARIABLES = {"temperature_2m": "Dry_Bulb", "dew_point_2m": "Dew_Point"}
ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch(lat: float, lon: float, start: str, end: str, lead: int, model: str) -> pd.DataFrame:
    hourly = ",".join(f"{v}_previous_day{lead}" for v in VARIABLES)
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "hourly": hourly, "models": model,
        "start_date": start, "end_date": end, "temperature_unit": "fahrenheit",
        "timezone": "GMT",
    })
    for attempt in range(1, 6):                              # the API is slow at times: retry with back-off
        try:
            with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=300) as response:
                payload = json.loads(response.read())
            break
        except Exception as error:                           # noqa: BLE001
            if attempt == 5:
                raise
            print(f"  retry {attempt} after {type(error).__name__}", flush=True)
            time.sleep(15 * attempt)
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.set_index("time")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--start", default="2024-12-01")   # forecasts are only used for the 2025 test origins
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--model", default="gfs_seamless")
    parser.add_argument("--out", type=Path, default=Path("data/raw/isone"))
    args = parser.parse_args()

    columns = {}
    for zone, (lat, lon) in ZONE_STATIONS.items():
        parts = []
        for month_start in pd.date_range(pd.Timestamp(args.start).replace(day=1), args.end, freq="MS"):
            month_end = min(month_start + pd.offsets.MonthEnd(0), pd.Timestamp(args.end))   # one request per month
            start = max(month_start, pd.Timestamp(args.start))
            parts.append(fetch(lat, lon, str(start.date()), str(month_end.date()), args.lead, args.model))
            time.sleep(1.0)
        frame = pd.concat(parts)
        for variable, name in VARIABLES.items():
            columns[f"{zone}_{name}"] = frame[f"{variable}_previous_day{args.lead}"]
        print(f"{zone}: {frame.notna().mean().min():.1%} of hours available", flush=True)

    out = pd.DataFrame(columns)
    out.index = out.index - pd.Timedelta(hours=5)          # UTC -> fixed UTC-05:00 clock
    out.index.name = "ts"
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"weather_forecasts_previous_day{args.lead}.csv"
    out.round(2).to_csv(path)
    print(f"wrote {path}: {len(out)} hours, {out.notna().all(axis=1).mean():.1%} complete")


if __name__ == "__main__":
    main()
