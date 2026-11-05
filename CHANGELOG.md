# Changelog

## v4 (November 2026)

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
