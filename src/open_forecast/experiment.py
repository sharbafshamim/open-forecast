# run everything under the same experiment setup
"""The common model-comparison harness."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
try:
    import torch
except ImportError:  # deep adapters unavailable; tree/naive runs still work
    torch = None

from .contracts import DatasetAudit
from .evaluation import metric_rows, prediction_rows
from .features import (
    SequenceScaler,
    build_sequence_partitions,
    build_tabular_frame,
    chronological_split,
    tabular_partitions,
)
from . import models as _models
from .models import TreeForecaster, naive_prediction
from .preprocessing import load_hourly_dataset
from .visualization import save_forecast_trace, save_mode_comparison


TREE_MODELS = {
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
}
DEEP_MODELS = {
    "lstm",
    "cross_attention_transformer",
    "patchtst",
}
from .foundation import FOUNDATION_MODELS as _FM

FOUNDATION_MODELS = set(_FM)
NAIVE_MODELS = {
    "persistence",
    "seasonal_naive",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    """Hash source/configuration files while excluding generated outputs and Git state."""
    digest = hashlib.sha256()
    excluded = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "results",
        "data",
    }

    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or excluded.intersection(
                path.relative_to(root).parts
            )
        ):
            continue

        digest.update(
            str(path.relative_to(root)).encode()
        )
        digest.update(path.read_bytes())

    return digest.hexdigest()


def _write_manifest(
    output: Path,
    config: dict,
    audit: DatasetAudit,
    split_manifest: dict,
    models: list[str],
    modes: list[str],
) -> None:
    repository_root = Path(__file__).resolve().parents[2]

    try:
        git_revision = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository_root),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "status",
                    "--porcelain",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )

        if dirty:
            git_revision = (
                f"{git_revision} "
                "(dirty working tree; verify source_tree_sha256)"
            )

    except (OSError, subprocess.CalledProcessError):
        git_revision = (
            "unavailable "
            "(repository not initialized or committed)"
        )

    raw_paths = config["dataset"].get("paths") or [config["dataset"]["path"]]
    data_paths = [Path(p) if Path(p).is_absolute() else repository_root / p for p in raw_paths]

    manifest = {
        "framework": "open-forecast-framework",
        "models": models,
        "feature_modes": modes,
        "datetime_audit": asdict(audit),
        "split": split_manifest,
        "feature_availability": {
            "weather_time": (
                "target-time weather proxy + known calendar"
            ),
            "weather_time_history": (
                "weather_time + historical consumption"
            ),
            "weather_time_history_weekly": (
                "weather_time_history + target-aligned weekly lags"
            ),
        },
        "config": config,
        "provenance": {
            "git_revision": git_revision,
            "source_tree_sha256": _source_tree_sha256(
                repository_root
            ),
            "raw_data_sha256": (
                _sha256_file(data_paths[0]) if len(data_paths) == 1
                else {p.name: _sha256_file(p) for p in data_paths}
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__ if torch is not None else "not installed",
        },
    }

    (output / "run_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        )
    )


def run_experiment(
    config: dict,
    models: list[str],
    modes: list[str],
    output: str | Path,
    epochs_override: int | None = None,
    split_builder=None,
    save_figures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all requested model/scenario combinations under one protocol.

    ``split_builder(common_index, horizon_hours) -> Split`` replaces the default
    fractional chronological split (used by the rolling-origin evaluation).
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)

    supported = TREE_MODELS | DEEP_MODELS | NAIVE_MODELS | FOUNDATION_MODELS
    if not set(models).issubset(supported):
        raise ValueError(
            f"Unsupported model requested: "
            f"{sorted(set(models) - supported)}"
        )

    if not set(modes).issubset(
        {"weather_time", "weather_time_history", "weather_time_history_weekly"}
    ):
        raise ValueError(
            f"Unsupported feature mode: {modes}"
        )

    dataset = config["dataset"]
    forecast = config["forecast"]
    training = dict(config["training"])

    if epochs_override is not None:
        training["epochs"] = epochs_override

    if dataset.get("format") == "isone_smd":
        from .isone import load_isone_smd
        hourly, audit = load_isone_smd(dataset["paths"], dataset["target_columns"])
    else:
        hourly, audit = load_hourly_dataset(
            dataset["path"],
            dataset["datetime_column"],
            dataset["target_columns"],
            dataset["weather_columns"],
        )

    # Build every requested mode's frame once, then use the INTERSECTION of
    # their valid origins as the common index. Different modes trim different
    # rows (weather_time only trims the tail via the target shift;
    # weather_time_history also trims the head via lag/rolling features), so
    # no single mode is guaranteed to be a superset of the others.
    mode_frames: dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    for mode in modes:
        mode_frames[mode] = build_tabular_frame(
            hourly,
            dataset["target_columns"],
            dataset["weather_columns"],
            mode,
            forecast["horizon_hours"],
            config["features"]["tree_lags"],
            config["features"]["rolling_windows"],
            config["features"].get("weekly_lags", [144, 168]),
        )

    common_index = mode_frames[modes[0]][0].index
    for mode in modes[1:]:
        common_index = common_index.intersection(
            mode_frames[mode][0].index
        )

    if len(common_index) == 0:
        raise ValueError(
            "No forecast origins are valid across every requested feature mode."
        )

    if split_builder is not None:
        split = split_builder(common_index, forecast["horizon_hours"])
    else:
        split = chronological_split(
            common_index,
            forecast["horizon_hours"],
            config["split"]["train_fraction"],
            config["split"]["validation_fraction"],
        )

    _write_manifest(
        output,
        config,
        audit,
        split.manifest(),
        models,
        modes,
    )

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    for mode in modes:
        tabular_x, tabular_y = mode_frames[mode]

        # Match every family to the origin set valid across every requested mode.
        tabular_x = tabular_x.loc[common_index]
        tabular_y = tabular_y.loc[common_index]
        tabs = tabular_partitions(
            tabular_x,
            tabular_y,
            split,
        )

        if set(models) & DEEP_MODELS:
            sequences = build_sequence_partitions(
                hourly,
                dataset["target_columns"],
                dataset["weather_columns"],
                mode,
                forecast["horizon_hours"],
                forecast["lookback_hours"],
                split,
                config["features"].get("weekly_lags", [144, 168]),
            )

            scaler = SequenceScaler()
            sequences["train"] = scaler.fit_transform(
                sequences["train"]
            )
            sequences["validation"] = scaler.transform(
                sequences["validation"]
            )
            sequences["test"] = scaler.transform(
                sequences["test"]
            )

        for model_name in models:
            # A persistence or seasonal-naive demand forecast consumes observed
            # consumption at the forecast origin. It is a valuable Mode B
            # reference, but it is not a valid exogenous-only Mode A method.
            if (
                model_name in NAIVE_MODELS
                and mode != "weather_time_history"
            ):
                continue

            # Foundation models always condition on target history, so they are
            # evaluated in the history mode only (chronos2 also gets the weather
            # proxy as covariates; chronos-bolt is univariate).
            if model_name in FOUNDATION_MODELS and mode != "weather_time_history":
                continue

            if model_name in FOUNDATION_MODELS:
                from .foundation import predict_load

                settings = config.get("foundation", {})
                validation_pred = predict_load(
                    model_name, hourly, tabs["validation"].x.index,
                    dataset["target_columns"], dataset["weather_columns"],
                    forecast["horizon_hours"], settings,
                ) if settings.get("score_validation", False) else tabs["validation"].y.to_numpy() * np.nan
                test_pred = predict_load(
                    model_name, hourly, tabs["test"].x.index,
                    dataset["target_columns"], dataset["weather_columns"],
                    forecast["horizon_hours"], settings,
                )
                validation_y = tabs["validation"].y.to_numpy()
                test_y = tabs["test"].y.to_numpy()
                validation_index = tabs["validation"].x.index
                test_index = tabs["test"].x.index

            elif model_name in NAIVE_MODELS:
                validation_pred = naive_prediction(
                    model_name,
                    hourly,
                    tabs["validation"].x.index,
                    dataset["target_columns"],
                    forecast["horizon_hours"],
                )
                test_pred = naive_prediction(
                    model_name,
                    hourly,
                    tabs["test"].x.index,
                    dataset["target_columns"],
                    forecast["horizon_hours"],
                )

                validation_y = (
                    tabs["validation"].y.to_numpy()
                )
                test_y = tabs["test"].y.to_numpy()
                validation_index = (
                    tabs["validation"].x.index
                )
                test_index = tabs["test"].x.index

            elif model_name in TREE_MODELS:
                forecaster = TreeForecaster(
                    model_name,
                    config["models"][model_name],
                    training["seed"],
                )
                validation_pred, test_pred = forecaster.fit_predict_lean(
                    tabs["train"], [tabs["validation"], tabs["test"]]
                )

                validation_y = (
                    tabs["validation"].y.to_numpy()
                )
                test_y = tabs["test"].y.to_numpy()
                validation_index = (
                    tabs["validation"].x.index
                )
                test_index = tabs["test"].x.index

            else:
                # Sequence and tabular origins are intentionally intersected,
                # preserving identical split semantics across model families.
                deep = _models.DeepForecaster(
                    model_name,
                    config["models"][model_name],
                    training,
                    training["seed"],
                ).fit(
                    sequences["train"],
                    sequences["validation"],
                )

                validation_pred = scaler.inverse_target(
                    deep.predict(
                        sequences["validation"]
                    )
                )
                test_pred = scaler.inverse_target(
                    deep.predict(
                        sequences["test"]
                    )
                )
                validation_y = scaler.inverse_target(
                    sequences["validation"].y
                )
                test_y = scaler.inverse_target(
                    sequences["test"].y
                )
                validation_index = (
                    sequences["validation"].index
                )
                test_index = sequences["test"].index

            scored = [("test", test_y, test_pred)]
            if not np.isnan(np.asarray(validation_pred, dtype=float)).all():
                scored.insert(0, ("validation", validation_y, validation_pred))
            all_metrics.extend(
                metric_rows(model_name, mode, name, truth, pred, dataset["target_columns"])
                for name, truth, pred in scored
            )

            all_predictions.append(
                prediction_rows(
                    model_name,
                    mode,
                    "test",
                    test_index,
                    test_y,
                    test_pred,
                    dataset["target_columns"],
                    forecast["horizon_hours"],
                )
            )

    metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )
    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics.to_csv(
        output / "metrics.csv",
        index=False,
    )
    metrics[
        metrics["split"] == "validation"
    ].to_csv(
        output / "validation_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        output / "predictions.csv",
        index=False,
    )

    if save_figures:
        save_mode_comparison(
            metrics,
            output / "figures",
        )
        save_forecast_trace(
            predictions,
            output / "figures",
            unit=dataset.get("unit", "dataset units"),
        )

    return metrics, predictions
