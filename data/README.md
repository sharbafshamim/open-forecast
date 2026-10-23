# Data

Raw data is not committed. Place the files below exactly at these paths; every
run records the SHA-256 of each raw file in its `run_manifest.json`.

| Path | Source |
|---|---|
| `data/raw/isone/2023_smd_hourly.xlsx`, `2024_…`, `2025_…` | *(main dataset)* ISO New England, *SMD Hourly Data* annual workbooks: ISO Express → Energy, Load and Demand Reports → Zonal Information (<https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info>). |
| `data/raw/power_consumption.csv` | *(secondary dataset)* Tetouan City power consumption (Kaggle version used in the original study; 10-min, 2017). Use the raw CSV unchanged; column names are normalised on load. |
| `data/raw/epf/{PJM,NP,DE,BE,FR}.csv` | EPF benchmark markets of Lago et al. (2021), Zenodo record 4624805. Downloaded by `python scripts/download_data.py`. |
| `data/raw/epf/nixtla_electricity_mirror.csv` | Optional price-only public mirror (BE, DE, FR, NP; no exogenous inputs), used only for pipeline previews. Downloaded by `python scripts/download_data.py --mirror`. |

## Declared data policies

* **ISO-NE daylight saving (load).** Hour-ending local time. 2024 onward follow the
  true clock (23-h March day, 25-h November day with `02X`); the 2023 file uses a
  legacy 24-rows-per-day convention (synthetic spring hour dropped, averaged autumn
  hour split and interpolated). Timestamps are hour-beginning on fixed UTC-05:00.
  See `src/open_forecast/isone.py`.
* **ISO-NE daylight saving (price).** Auctions clear per local market day, so price
  days use 24 local wall-clock hours (missing spring hour interpolated, `02`/`02X`
  averaged). See `load_isone_price` in `src/open_forecast/price.py`.
* **Excluded columns.** `DA_Demand` is never used: it is cleared by the same
  day-ahead auction whose price is being forecast.
* **Weather.** Observed target-time weather is used as a *declared proxy* for a
  weather forecast in both the load and the ISO-NE price experiments.
* **Tetouan units.** The UCI record (DOI 10.24432/C5B034) leaves the unit of every
  variable blank. Values are used as published and labelled "dataset units";
  cross-dataset comparisons use `nrmse_pct`. Do not convert or relabel them as
  W or kW in code, figures or text.
