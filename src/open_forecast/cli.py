# define the command line runner
"""Command-line entry point for a reproducible forecasting experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled feature-availability forecasting experiments."
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run = subparsers.add_parser(
        "run",
        help="run models through the common experiment harness",
    )
    run.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    run.add_argument(
        "--models",
        nargs="+",
        required=True,
    )
    run.add_argument(
        "--modes",
        nargs="+",
        required=True,
    )
    run.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    run.add_argument(
        "--epochs",
        type=int,
        help="override deep-model epochs for a smoke test",
    )
    for sub in (run,):
        _add_common(sub)

    price = subparsers.add_parser(
        "price",
        help="day-ahead price forecasting with rolling recalibration",
    )
    price.add_argument("--config", type=Path, required=True)
    price.add_argument("--models", nargs="+", default=["naive", "lear", "hist_gradient_boosting", "extra_trees"])
    price.add_argument("--information-sets", nargs="+", default=["price_only", "price_exogenous"])
    price.add_argument("--output", type=Path, required=True)

    rolling = subparsers.add_parser(
        "rolling",
        help="rolling-origin evaluation of the load harness over several test windows",
    )
    rolling.add_argument("--config", type=Path, required=True)
    rolling.add_argument("--models", nargs="+", required=True)
    rolling.add_argument("--modes", nargs="+", required=True)
    rolling.add_argument("--output", type=Path, required=True)
    rolling.add_argument("--epochs", type=int)
    _add_common(rolling)

    args = parser.parse_args()

    if args.command == "price":
        from .price import run_price_experiment
        config = yaml.safe_load(args.config.read_text())
        root = Path(__file__).resolve().parents[2]
        metrics = run_price_experiment(config, args.models, args.information_sets, args.output, root)
        print(metrics.to_string(index=False))
        return

    if args.command in ("rolling", "run"):
        base = yaml.safe_load(args.config.read_text())
        if args.weather_forecast:
            base["dataset"]["weather_forecast_csv"] = str(args.weather_forecast)
        seeds = args.seeds or [None]
        for seed in seeds:
            config = yaml.safe_load(yaml.safe_dump(base))
            output = args.output
            if seed is not None:          # one sub-folder per training seed
                config["training"]["seed"] = seed
                output = args.output / f"seed_{seed}"
            if args.command == "rolling":
                from .rolling import run_rolling
                summary = run_rolling(config, args.models, args.modes, output, args.epochs)
                print(summary.to_string(index=False))
            else:
                metrics, _ = run_experiment(config, args.models, args.modes, output, args.epochs)
                print(metrics[(metrics["split"] == "test") & (metrics["scope"] == "aggregate")]
                      .sort_values(["feature_mode", "rmse"]).to_string(index=False))


def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--seeds", nargs="+", type=int,
                     help="repeat the run for each training seed (outputs in <output>/seed_<s>)")
    sub.add_argument("--weather-forecast", type=Path,
                     help="CSV of archived day-ahead weather forecasts used for the test origins")


if __name__ == "__main__":
    main()
