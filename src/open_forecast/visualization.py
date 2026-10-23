# save the comparison plots
"""Consistent figures for model/scenario comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_mode_comparison(metrics: pd.DataFrame, output: Path) -> None:
    """Plot test R² for every non-aggregate zone and feature scenario."""
    view = metrics[
        (metrics["split"] == "test") & (metrics["scope"] != "aggregate")
    ]
    if view.empty:
        return

    labels = view["model"] + "\n" + view["feature_mode"]
    figure, axis = plt.subplots(
        figsize=(max(9, len(view) * 0.55), 4.8)
    )

    for zone in sorted(view["scope"].unique()):
        slice_ = view[view["scope"] == zone]
        axis.plot(
            [
                labels.iloc[list(view.index).index(index)]
                for index in slice_.index
            ],
            slice_["r2"],
            "o-",
            label=zone,
        )

    axis.axhline(0, color="black", linewidth=0.75, alpha=0.5)
    axis.set_ylabel(r"Held-out $R^2$ at $t+24$")
    axis.set_xlabel("Model and feature-availability scenario")
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(output / "mode_comparison.png", dpi=220)
    plt.close(figure)


def save_forecast_trace(
    predictions: pd.DataFrame,
    output: Path,
    hours: int = 168,
    unit: str = "dataset units",
) -> None:
    """Save a last-week trace for the first model/scenario pair in the run."""
    test = predictions[predictions["split"] == "test"]
    if test.empty:
        return

    selected = test[["model", "feature_mode"]].drop_duplicates().iloc[0]
    selected = test[
        (test["model"] == selected["model"])
        & (test["feature_mode"] == selected["feature_mode"])
    ]

    figure, axes = plt.subplots(
        len(selected["zone"].unique()),
        1,
        figsize=(8, 6),
        sharex=True,
    )
    axes = axes if hasattr(axes, "__len__") else [axes]

    for axis, zone in zip(axes, sorted(selected["zone"].unique())):
        subset = selected[selected["zone"] == zone].tail(hours)
        axis.plot(
            subset["target_time"],
            subset["actual"],
            label="Actual",
            linewidth=1.5,
        )
        axis.plot(
            subset["target_time"],
            subset["predicted"],
            label="Predicted",
            linewidth=1.2,
        )
        axis.set_ylabel(f"{zone} ({unit})")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    axes[-1].set_xlabel("Target time")
    figure.suptitle(
        f"{selected['model'].iloc[0]} — {selected['feature_mode'].iloc[0]}",
        y=0.995,
    )
    figure.tight_layout()
    figure.savefig(output / "forecast_trace.png", dpi=220)
    plt.close(figure)
