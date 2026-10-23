"""Rolling-origin evaluation: the same harness, many consecutive test windows.

A single chronological split evaluates every model on one test window. When the
window is atypical (e.g. unusually regular weeks) the model ranking can flip.
This module re-runs the unchanged experiment harness on consecutive test
windows with an expanding training set:

    train      = origins  < validation_start - H          (purged)
    validation = [test_start - validation_days, test_start - H)   (early stopping)
    test       = [test_start, test_start + window_days)

Every fold keeps the H-hour embargo between partitions, so no target used for
fitting overlaps a validation or test target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import Split
from .experiment import run_experiment


def make_fold_builder(test_start: pd.Timestamp, window_days: int, validation_days: int):
    def build(index: pd.DatetimeIndex, horizon: int) -> Split:
        h = pd.Timedelta(hours=horizon)
        test_end = test_start + pd.Timedelta(days=window_days)
        validation_start = test_start - pd.Timedelta(days=validation_days)
        train = index[index < validation_start - h]
        validation = index[(index >= validation_start) & (index < test_start - h)]
        test = index[(index >= test_start) & (index < test_end)]
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError(f"Empty partition for fold starting {test_start}")
        return Split(train, validation, test, horizon, horizon, horizon)

    return build


def run_rolling(config: dict, models: list[str], modes: list[str], output: str | Path,
                epochs_override: int | None = None) -> pd.DataFrame:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rolling = config["rolling"]
    starts = pd.date_range(rolling["first_test_start"], rolling["last_test_start"],
                           freq=f"{rolling['window_days']}D")
    all_metrics = []
    for fold, start in enumerate(starts):
        fold_dir = output / f"fold_{fold:02d}_{start.date()}"
        if (fold_dir / "metrics.csv").exists():
            metrics = pd.read_csv(fold_dir / "metrics.csv")
        else:
            metrics, _ = run_experiment(
                config, models, modes, fold_dir, epochs_override,
                split_builder=make_fold_builder(start, rolling["window_days"], rolling["validation_days"]),
                save_figures=False,
            )
        metrics = metrics[metrics["split"] == "test"].copy()
        metrics.insert(0, "fold", fold)
        metrics.insert(1, "test_start", str(start.date()))
        all_metrics.append(metrics)
        print(f"fold {fold} ({start.date()}) done", flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output / "rolling_metrics.csv", index=False)

    agg = metrics[metrics["scope"] == "aggregate"]
    agg = agg.assign(rank=agg.groupby("fold")["rmse"].rank())
    summary = (
        agg.groupby(["model", "feature_mode"])
        .agg(folds=("fold", "nunique"), mean_rmse=("rmse", "mean"),
             median_rmse=("rmse", "median"), mean_rank=("rank", "mean"),
             wins=("rank", lambda r: int((r == 1).sum())))
        .reset_index()
        .sort_values("mean_rank")
    )
    summary.to_csv(output / "rolling_summary.csv", index=False)
    return summary
