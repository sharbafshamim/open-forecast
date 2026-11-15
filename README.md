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

* **Information sets are declared as available data**, not feature lists, and every
  model family may use any subset of them:
  * Load: **A** calendar + weather up to the target hour; **B** A + load observed in
    the last 24 h; **B+** A + all load observed up to the origin (weekly lags,
    the weekly naive forecast and the 512-h Chronos contexts belong here).
  * Price: `price_only` (prices of D-1 and earlier) and `price_exogenous`
    (+ inputs known at the 10:00 bidding gate of D-1: published day-ahead
    forecasts for the EPF markets; for ISO-NE a declared weather proxy at D and
    observed load of D-2, D-7 and of D-1 up to hour-ending 09:00).
* **Leakage audit.** `tests/test_information_sets.py` perturbs every datum outside
  a set's window (future load, weather after the target hour, undeclared load) and
  fails if any input of trees, deep models or foundation models changes; a gate
  test does the same for the price inputs.
* **No look-ahead.** Chronological splits with an H-hour purge; every trained model
  is refit before each rolling test window (load) or recalibrated weekly (price).
* **Same origins for every model family.**
* **Rolling-origin evaluation** on both datasets (ISO-NE: 13 windows of 2025).
* **Real weather forecasts** can replace the observed-weather proxy at test time.
* **Significance.** Diebold-Mariano (HLN, Newey-West one-week window, daily
  variant, Holm adjustment) for load; multivariate DM (i.i.d. and Newey-West) for
  price; Wilcoxon across rolling windows.
* **Provenance.** Manifests with config, splits, data and code hashes, git revision
  and package versions; compressed test-set forecasts are committed.

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

# Load: rolling-origin evaluation (13 x 28-day windows over 2025, refit before each)
python -m open_forecast.cli rolling --config configs/isone_rolling.yaml \
  --models persistence hist_gradient_boosting --modes weather_time_history \
  --output results/isone_rolling_example

# Price: day-ahead protocol with rolling recalibration
python -m open_forecast.cli price --config configs/isone_price.yaml \
  --models naive lear hist_gradient_boosting chronos2 --output results/price_example

# Everything in the paper, then tables and figures
bash scripts/reproduce.sh
python analysis/paper_tables.py results   # result tables and figures -> results/paper/
python analysis/eda_figures.py            # data figures -> results/eda/
```

Foundation models run on `OPEN_FORECAST_DEVICE` (`cpu`, `cuda` or `mps`).

## Results

`python analysis/paper_tables.py results` rebuilds every table of the paper in
`results/paper/`. The v5 runs (monthly refits, equal information, bidding gate,
weather forecasts) are produced by `scripts/reproduce.sh` or
`notebooks/revision_runs.ipynb`; results of the superseded v4 protocol are kept in
`results/archive_v4/` with an explanation.

**ISO-NE load, 13 monthly-refit windows of 2025 (RMSE in MW, 8,736 origins)**

| Model | Set | RMSE, observed weather | nRMSE (%) | RMSE, archived GFS forecasts |
|---|---|---|---|---|
| LSTM | B+ | **89.8** | **5.44** | **113.8** |
| LSTM | B | 90.2 | 5.47 | 114.8 |
| HistGradientBoosting | B | 96.1 | 5.82 | 115.1 |
| Cross-attention Transformer | B+ | 99.8 | 6.05 | – |
| PatchTST | B+ | 105.5 | 6.39 | – |
| Chronos-2, zero-shot | B+ | 108.9 | 6.60 | 128.3 |
| Random Forest | A | 115.7 | 7.01 | 141.8 |
| Chronos-Bolt small, zero-shot | B+ | 189.2 | 11.46 | – |
| Persistence | B | 208.6 | 12.63 | 208.6 |

With observed weather the LSTM is best, but ahead of gradient boosting in only one
zone after the Holm adjustment; with real day-ahead weather forecasts the two are tied.

**ISO-NE day-ahead price, 2025 (bidding-gate inputs, weekly refits)**

| Model | Inputs | MAE ($/MWh) | rMAE |
|---|---|---|---|
| Chronos-2, zero-shot | price + exogenous | **10.28** | **0.518** |
| Chronos-2 + LEAR (equal weights) | price + exogenous | 10.54 | 0.531 |
| LEAR | price + exogenous | 12.45 | 0.627 |
| HistGradientBoosting | price + exogenous | 12.51 | 0.631 |
| Chronos-2, zero-shot | price | 12.40 | 0.625 |
| LEAR | price | 13.16 | 0.663 |
| Naive | price | 19.84 | 1.000 |

**EPF markets: rMAE relative to the naive forecast (lower is better)**

| Model | Inputs | PJM | NP | DE | BE | FR |
|---|---|---|---|---|---|---|
| Chronos-2, zero-shot | price + exogenous | 0.613 | **0.488** | **0.394** | **0.704** | **0.621** |
| Chronos-2 + LEAR (equal weights) | price + exogenous | **0.588** | 0.519 | 0.400 | 0.705 | 0.622 |
| LEAR (this repo, one 1,456-day window) | price + exogenous | 0.636 | 0.641 | 0.494 | 0.792 | 0.729 |
| LEAR, one 1,456-day window (published, Lago et al. 2021) | price + exogenous | 0.639 | 0.629 | 0.496 | 0.809 | 0.735 |
| LEAR ensemble (published) | price + exogenous | 0.622 | 0.549 | 0.449 | 0.749 | 0.668 |
| DNN ensemble (published) | price + exogenous | 0.591 | 0.532 | 0.425 | 0.716 | 0.649 |

**Tetouan, 11 rolling windows:** zero-shot Chronos-Bolt small has mean rank 2.0,
wins 7 windows and beats the weekly naive forecast in all 11 (Wilcoxon p = 0.001);
the best tree ensemble (random forest, B+) is level with the naive forecast (p = 0.41).

The LEAR implementation reproduces the published single-window LEAR to within
0.017 rMAE. The five markets are fev-bench tasks, which the Chronos-2 authors
report were not used to train it.

> Units: ISO-NE loads are in MW and prices in $/MWh. The Tetouan source (UCI
> DOI 10.24432/C5B034) states no unit for consumption, so Tetouan errors are
> reported in dataset units and as `nrmse_pct` (RMSE as % of mean load).

## Notebooks

| Notebook | What it does |
|---|---|
| `notebooks/forecasting_analysis.ipynb` | v4 step-by-step walkthrough (superseded protocol, kept for reference) |
| `notebooks/chronos_only.ipynb` | v4 Chronos-only runs (superseded; use `revision_runs.ipynb`) |
| `notebooks/revision_runs.ipynb` | v5 runs that need a GPU or internet: ISO-NE monthly refits, Chronos, weather forecasts, price with the bidding gate, published EPF comparison |
| `notebooks/nov15_runs.ipynb` | executed copy of `revision_runs.ipynb` with all outputs: the exact run (15 Nov 2026) behind the paper's results |
| `notebooks/multi_seed_and_dm.ipynb` | v4 five-seed study (superseded protocol, kept for reference) |

## Repository layout

```
src/open_forecast/
  preprocessing.py   Tetouan loader (explicit timestamp parsing, column normalisation)
  isone.py           ISO-NE SMD loader with declared daylight-saving policy
  features.py        information-set feature builders, purged chronological split
  experiment.py      common load harness (all model families, manifests)
  rolling.py         rolling-origin evaluation on top of the harness
  price.py           day-ahead price protocol (bidding gate), LEAR / trees, rMAE
  significance.py    DM tests (Newey-West, daily, Holm), multivariate DM, window tests
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

## License

MIT (see `LICENSE`).
