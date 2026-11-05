# Open Forecast

**A leakage-audited benchmark for day-ahead electricity load and price forecasting.**
Tree ensembles, deep sequence models and zero-shot time-series foundation models
are compared under one harness, with explicitly declared information sets,
horizon-purged splits and significance tests.

| Task | Models |
|---|---|
| Load, 24 h ahead (hourly origins) | persistence, seasonal naive, Random Forest, Extra Trees, HistGradientBoosting, LSTM, Cross-Attention Transformer, PatchTST, Chronos-Bolt, Chronos-2 |
| Day-ahead price (24 prices of day D issued on D-1) | naive, LEAR, HistGradientBoosting, Extra Trees, Chronos-Bolt, Chronos-2 |

## Datasets and their roles

| Role | Dataset | Used for |
|---|---|---|
| **Main** | **ISO New England** SMD hourly, 2023–2025 (8 load zones + system), test year 2025 | load forecasting (all model families) and day-ahead price forecasting |
| Price benchmark | EPF markets of Lago et al. (2021): PJM, Nord Pool, EPEX-DE/BE/FR | external validity of the price results on the standard open benchmark |
| Secondary | Tetouan, Morocco, 2017 (2 zones, 10-min data aggregated to hourly) | robustness on a small single-year system and the rolling-origin analysis |

ISO-NE carries the main claims. Tetouan is kept because it is widely used in the
literature (comparability) and because it shows how a single test window can
reverse model rankings on small datasets.

## Design principles

* **Declared information sets.** Every feature is tagged with when it becomes known.
  * Load modes: **A** `weather_time` (target-time weather proxy + calendar),
    **B** `weather_time_history` (+ consumption history at the origin),
    **B+** `weather_time_history_weekly` (+ the target-aligned weekly lags the
    seasonal-naive benchmark uses).
  * Price sets: `price_only` (prices D-1, D-2, D-3, D-7 + weekday) and
    `price_exogenous` (+ exogenous inputs at D, D-1, D-7 — published day-ahead
    forecasts for the EPF markets; for ISO-NE a declared weather proxy at D and
    observed load at D-1/D-7 only). Market-cleared `DA_Demand` is never used.
* **No look-ahead.** Chronological splits with an H-hour purge/embargo between
  partitions; price models are recalibrated on rolling windows using only data
  strictly before the forecast day.
* **Same origins for every model family.** Tabular, sequence and foundation
  models are scored on the identical set of forecast origins.
* **Rolling-origin evaluation.** The unchanged harness re-run on consecutive
  28-day test windows, so conclusions do not hinge on one test period.
* **Significance.** Diebold–Mariano tests (HLN correction, Newey–West variance)
  for load; multivariate DM test (Ziel & Weron 2018) for price.
* **Provenance.** Each run writes a manifest with the config, split boundaries,
  SHA-256 of every raw file and of the source tree, and package versions.

## Install

```bash
pip install -e ".[dev]"          # trees, naive, price models, tests
pip install -e ".[deep]"         # + LSTM / Transformer / PatchTST (PyTorch)
pip install -e ".[foundation]"   # + Chronos-Bolt / Chronos-2 (weights from Hugging Face)
pip install -e ".[all]"          # everything
pytest -q
```

## Data

Raw data is not committed. See [`data/README.md`](data/README.md) for sources,
expected paths and the declared daylight-saving policies.

```bash
python scripts/download_data.py   # EPF benchmark markets (Zenodo 4624805)
# place ISO-NE SMD workbooks and the Tetouan CSV manually (data/README.md)
```

## Usage

```bash
# Load: one chronological split, all modes
python -m open_forecast.cli run --config configs/isone.yaml \
  --models seasonal_naive hist_gradient_boosting chronos2 \
  --modes weather_time weather_time_history weather_time_history_weekly \
  --output results/isone_example

# Load: rolling-origin evaluation (11 x 28-day windows)
python -m open_forecast.cli rolling --config configs/tetouan_rolling.yaml \
  --models seasonal_naive random_forest --modes weather_time_history_weekly \
  --output results/tetouan_rolling_example

# Price: day-ahead protocol with rolling recalibration
python -m open_forecast.cli price --config configs/isone_price.yaml \
  --models naive lear hist_gradient_boosting chronos2 --output results/price_example

# Everything in the paper, then tables and figures
bash scripts/reproduce.sh
python analysis/summarize.py results      # result tables and figures -> results/paper/
python analysis/eda_figures.py            # data figures -> results/eda/
```

Foundation models run on `OPEN_FORECAST_DEVICE` (`cpu`, `cuda` or `mps`).

## Results

All tables of the paper are rebuilt from the committed results with
`python analysis/paper_tables.py results` (written to `results/paper/`).
Per-point predictions are not committed; the scripts regenerate them.

**ISO-NE load, test year 2025: aggregate RMSE over 8 zones (MW)**

| Model | A: weather + calendar | B: + history | B+: + weekly lags |
|---|---|---|---|
| **LSTM** (mean of 5 seeds for B, B+) | **109.0** | **96.6** | **96.8** |
| Gradient boosting | 130.4 | 102.2 | 103.1 |
| Chronos-2 with weather covariates, zero-shot | – | 108.9 | – |
| Cross-attention Transformer (5 seeds) | 125.4 | 112.3 | 111.5 |
| Extra trees | 129.9 | 115.0 | 117.5 |
| PatchTST (5 seeds) | 136.2 | 120.8 | 116.6 |
| Random forest | 133.8 | 119.6 | 122.9 |
| Chronos-Bolt small / base, zero-shot | – | 189.7 / 187.6 | – |
| Persistence | – | 208.8 | – |
| Weekly seasonal naive | – | 305.1 | – |

All ten LSTM runs beat gradient boosting; the LSTM is significantly more accurate
in 4–7 of the 8 zones depending on the seed (`results/deep/`).

**Day-ahead price: rMAE relative to the naive forecast (lower is better)**

| Model | Inputs | ISO-NE | PJM | NP | DE | BE | FR |
|---|---|---|---|---|---|---|---|
| **Chronos-2, zero-shot** | **price + exogenous** | **0.512** | **0.613** | **0.488** | **0.394** | **0.704** | **0.621** |
| LEAR | price + exogenous | 0.615 | 0.636 | 0.641 | 0.494 | 0.792 | 0.729 |
| Chronos-2, zero-shot | price only | 0.625 | 0.711 | 0.612 | 0.645 | 0.771 | 0.716 |
| Gradient boosting | price + exogenous | 0.666 | 0.667 | 0.717 | 0.513 | 0.807 | 0.792 |
| LEAR | price only | 0.663 | 0.726 | 0.708 | 0.654 | 0.804 | 0.760 |

Chronos-2 with exogenous inputs is significantly more accurate than every other
model in all six markets (multivariate DM test, p ≤ 0.019). It receives exactly
the inputs declared as known on day D, the same ones LEAR receives.

**Tetouan.** On the conventional Nov–Dec test window the weekly seasonal naive
forecast (RMSE 1,216 dataset units) beats every trained model; only zero-shot
Chronos-Bolt small beats it (972). Across 11 rolling 28-day windows Chronos-Bolt
small has the best mean rank (1.6) and wins 7 windows; the naive forecast fails in
the windows around Ramadan/Eid al-Fitr and Eid al-Adha
(`results/paper/figure_tetouan_rolling.png`).

> Units: ISO-NE loads are in MW and prices in $/MWh. The Tetouan source (UCI
> DOI 10.24432/C5B034) states no unit for consumption, so Tetouan errors are
> reported in dataset units and as `nrmse_pct` (RMSE as % of mean load).

## Notebooks

| Notebook | What it does |
|---|---|
| `notebooks/forecasting_analysis.ipynb` | the whole study step by step, with results under each step |
| `notebooks/chronos_only.ipynb` | only the zero-shot Chronos runs (needs internet the first time) |
| `notebooks/multi_seed_and_dm.ipynb` | deep models with 5 seeds and the per-zone DM tests against gradient boosting and Chronos-2 |

## Repository layout

```
src/open_forecast/
  preprocessing.py   Tetouan loader (explicit timestamp parsing, column normalisation)
  isone.py           ISO-NE SMD loader with declared daylight-saving policy
  features.py        information-set feature builders, purged chronological split
  experiment.py      common load harness (all model families, manifests)
  rolling.py         rolling-origin evaluation on top of the harness
  price.py           day-ahead price protocol, LEAR / trees, rMAE, multivariate DM
  foundation.py      zero-shot Chronos-Bolt / Chronos-2 adapters (load and price)
  models/            tree, naive and PyTorch adapters
  cli.py             `run`, `rolling`, `price` commands
configs/             one YAML per dataset / protocol
analysis/            Diebold–Mariano tests (load), result tables/figures, paper tables, data figures
notebooks/           step-by-step notebooks (full study, Chronos only, multi-seed study)
scripts/             data download, full reproduction
tests/               unit tests (no GPU, no downloads; foundation-model alignment
                     is tested with a fake pipeline that must equal seasonal naive)
```

## Data credits

* ISO New England Inc., SMD Hourly Data (ISO Express). Data are used under ISO-NE's
  terms of use and are not redistributed here.
* Salam, A., El Hibaoui, A. (2018). Comparison of machine learning algorithms for
  the power consumption prediction: Case study of Tetouan city. *6th International
  Renewable and Sustainable Energy Conference (IRSEC)*. Dataset: UCI Machine
  Learning Repository, "Power Consumption of Tetouan City".
* Lago, J. et al. (2021), EPF benchmark datasets, Zenodo record 4624805.

## References

* Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). Forecasting day-ahead
  electricity prices: A review of state-of-the-art algorithms, best practices and
  an open-access benchmark. *Applied Energy* 293, 116983.
* Ziel, F., Weron, R. (2018). Day-ahead electricity price forecasting with
  high-dimensional structures: Univariate vs. multivariate modeling frameworks.
  *Energy Economics* 70, 396–420.
* Uniejewski, B., Weron, R., Ziel, F. (2018). Variance stabilizing transformations
  for electricity spot price forecasting. *IEEE Trans. Power Systems* 33(2).
* Ansari, A. F. et al. (2024). Chronos: Learning the language of time series. *TMLR*.
