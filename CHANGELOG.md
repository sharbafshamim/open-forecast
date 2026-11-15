# Changelog

## v5 (November 2026): response to an external review

* **Final results committed** (`results/`, tables in `results/paper/`): all v5
  runs, including the ISO-NE runs with archived weather forecasts, price with the
  bidding gate on ISO-NE and the five EPF markets, and the Chronos-2 + LEAR
  combination. Per-window fold folders are left out; compressed test-set
  forecasts are included so every significance test can be re-run.
* `scripts/fetch_weather_forecasts.py` requests one month at a time with retries
  (year-long requests timed out).

* **Fix (leakage, ISO-NE price):** observed load is now used only where it is
  known at the 10:00 bidding gate of D-1 (D-2, D-7 and D-1 hours ending by 09:00);
  previously all of D-1 was used. Chronos-2 receives the load covariate delayed
  to the gate. Test `test_isone_price_inputs_respect_the_bidding_gate`.
* **Information sets are defined by available data**, not by feature lists, and
  verified for every model family by perturbation tests
  (`tests/test_information_sets.py`). Trees now also see the load at the origin
  (lag 0, lags 0-23 h for set B) and summaries of the 24-h weather path; the
  weekly naive forecast and the Chronos models are labelled B+.
* **Load models are refit before every test window**: ISO-NE 2025 is evaluated
  in 13 rolling 28-day windows (`configs/isone_rolling.yaml`).
* **Archived weather forecasts** can replace observed weather at test time
  (`--weather-forecast`, `scripts/fetch_weather_forecasts.py`).
* **Statistics:** Newey-West window of one week for load DM tests, a daily
  aggregated DM test, Holm adjustment, Newey-West multivariate DM for price
  (`dm_pvalues_nw.csv`), Wilcoxon test across rolling windows
  (`src/open_forecast/significance.py`).
* **Metrics:** mean zonal nRMSE, MAPE, peak-hour RMSE; price peak and spike MAE.
* **Equal recalibration** for ISO-NE price: every model refits weekly.
* **Reproducibility:** `reproduce.sh` and `analysis/paper_tables.py` use the same
  folders; seeds and weather forecasts are CLI options (`--seeds`,
  `--weather-forecast`); compressed test-set forecasts are written and committed;
  the price forecast cache is keyed by a hash of the configuration.
* **Forecast combination:** `analysis/price_ensemble.py` averages Chronos-2 and LEAR
  (equal weights, no fitting) and tests it against both members and the published
  Lago et al. ensembles.
* `notebooks/revision_runs.ipynb` runs everything that needs a GPU or internet.

## v4 (5 November 2026)

* **Fix:** sequence models with weekly lags (information set B+) received NaN
  inputs for the earliest training origins, whose 24-h lookback reaches before
  the first available 168-h lag. Incomplete training/validation samples are now
  dropped, and incomplete test samples raise an error (`features.build_sequence_partitions`).
  New unit test `test_weekly_sequences_drop_incomplete_training_samples`.
* Results for all model families: deep models (`results/deep/`, including a
  five-seed study), zero-shot Chronos-Bolt and Chronos-2 for load and price, and
  the five official EPF benchmark markets.
* `analysis/paper_tables.py` rebuilds Tables 3-6 and Figure 6 of the paper.
* `notebooks/`: step-by-step notebooks for the full study, the Chronos runs and
  the multi-seed study.
* Rolling-origin results of deep models with weekly lags, computed before the
  fix, were removed.
