"""Build the paper's result tables and figures from the results/ folder.

Usage:  python analysis/summarize.py [results_dir]

Writes results/paper/: CSV tables (load, price, Tetouan rolling) and two
figures (Tetouan rolling RMSE per test window; price rMAE per market).
Missing runs are skipped, so the script can be re-run as results arrive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = ROOT / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# Reference categorical palette (light mode), fixed order; text stays neutral.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
LEGACY = {"rmse_watts": "rmse", "mae_watts": "mae", "actual_watts": "actual", "predicted_watts": "predicted"}


def read_metrics(path) -> pd.DataFrame:
    """Read a metrics file; results written before v0.3 used *_watts column names."""
    return pd.read_csv(path).rename(columns=LEGACY)


MODE_LABEL = {"weather_time": "A (weather+calendar)", "weather_time_history": "B (+history)",
              "weather_time_history_weekly": "B+ (+weekly lags)"}


def load_table() -> None:
    """Aggregate and per-zone test RMSE for every ISO-NE / Tetouan load run."""
    frames = []
    for dataset in ("isone", "tetouan_v2"):
        for run in sorted(ROOT.glob(f"{dataset}_*/metrics.csv")):
            if "rolling" in run.parent.name:
                continue
            m = read_metrics(run)
            m = m[m["split"] == "test"]
            m.insert(0, "dataset", "ISO-NE" if dataset == "isone" else "Tetouan")
            frames.append(m)
    if not frames:
        return
    m = pd.concat(frames).drop_duplicates(["dataset", "model", "feature_mode", "scope"])
    m["mode"] = m["feature_mode"].map(MODE_LABEL)
    agg = m[m["scope"] == "aggregate"].pivot_table(index=["dataset", "model"], columns="mode",
                                                   values="rmse")
    agg.round(1).to_csv(OUT / "table_load_aggregate_rmse.csv")
    m.to_csv(OUT / "table_load_all_scopes.csv", index=False)
    print("\nLOAD - aggregate test RMSE (ISO-NE: MW; Tetouan: dataset units, no unit stated by the source)\n", agg.round(1).to_string())


def price_table() -> None:
    rows = []
    for run in sorted(ROOT.glob("*price*/price_metrics.csv")):
        market = run.parent.name.replace("price_", "").replace("_price", "").replace("epf_", "").replace("_mirror", " (price-only mirror)")
        m = pd.read_csv(run)
        m.insert(0, "market", market.upper())
        rows.append(m)
    if not rows:
        return
    m = pd.concat(rows)
    m.to_csv(OUT / "table_price_all.csv", index=False)
    wide = m.pivot_table(index=["model", "information_set"], columns="market", values="rmae")
    wide.round(3).to_csv(OUT / "table_price_rmae.csv")
    print("\nPRICE - rMAE vs epftoolbox naive (lower is better)\n", wide.round(3).to_string())

    # figure: rMAE per market, one marker row per model (dot plot, no dual axes)
    best = m[m["model"] != "naive"].copy()
    models = list(dict.fromkeys(best["model"]))[:8]
    markets = list(dict.fromkeys(best["market"]))
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(markets) + 1.4))
    for i, model in enumerate(models):
        for info, offset, filled in (("price_only", -0.12, False), ("price_exogenous", 0.12, True)):
            sub = best[(best["model"] == model) & (best["information_set"] == info)]
            sub = sub.set_index("market").reindex(markets)
            if sub["rmae"].isna().all():
                continue
            ax.plot(sub["rmae"], [k + offset for k in range(len(markets))], "o", ms=7,
                    color=SERIES[i], markerfacecolor=SERIES[i] if filled else "white",
                    markeredgewidth=1.8, label=f"{model} ({'price + exog.' if filled else 'price only'})")
    ax.axvline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(markets)), markets)
    ax.set_ylim(-0.6, len(markets) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("rMAE (MAE relative to the naive forecast; 1.0 = naive)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(OUT / "figure_price_rmae.png")
    fig.savefig(OUT / "figure_price_rmae.pdf")
    plt.close(fig)


def rolling_table() -> None:
    frames = [read_metrics(p) for p in ROOT.glob("tetouan_rolling*/rolling_metrics.csv")]
    if not frames:
        return
    m = pd.concat(frames)
    agg = m[m["scope"] == "aggregate"].drop_duplicates(["fold", "model", "feature_mode"]).copy()
    agg["rank"] = agg.groupby("fold")["rmse"].rank()
    summary = (agg.groupby(["model", "feature_mode"])
               .agg(windows=("fold", "nunique"), mean_rmse=("rmse", "mean"),
                    mean_rank=("rank", "mean"), wins=("rank", lambda r: int((r == 1).sum())))
               .sort_values("mean_rank"))
    summary.round(2).to_csv(OUT / "table_tetouan_rolling.csv")
    print("\nTETOUAN rolling-origin (28-day windows)\n", summary.round(1).to_string())

    # figure: RMSE per window for the naive benchmarks and the best mode of each learner
    keep = [("seasonal_naive", "weather_time_history"), ("persistence", "weather_time_history")]
    learners = summary.reset_index()
    learners = learners[~learners["model"].isin(["seasonal_naive", "persistence"])]
    keep += list(learners.groupby("model").head(1)[["model", "feature_mode"]].itertuples(index=False, name=None))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, (model, mode) in enumerate(keep[:6]):
        sub = agg[(agg["model"] == model) & (agg["feature_mode"] == mode)].sort_values("fold")
        label = model if model in ("seasonal_naive", "persistence") else f"{model} ({MODE_LABEL[mode][:2].strip()})"
        ax.plot(pd.to_datetime(sub["test_start"]), sub["rmse"], marker="o", ms=4,
                color=SERIES[i], label=label)
    import matplotlib.dates as mdates

    # Holiday periods that break weekly regularity in Morocco in 2017
    for start, end, text in (("2017-05-27", "2017-06-27", "Ramadan / Eid al-Fitr"),
                             ("2017-08-31", "2017-09-03", "Eid al-Adha")):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=GRID, alpha=0.8, lw=0, zorder=0)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1], text, fontsize=7, color=MUTED, va="top")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Aggregate RMSE, 24-h ahead\n(dataset units)")
    ax.set_xlabel("Start of 28-day test window (2017); shaded = holiday periods")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(OUT / "figure_tetouan_rolling.png")
    fig.savefig(OUT / "figure_tetouan_rolling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    load_table()
    price_table()
    rolling_table()
    print(f"\nWrote {OUT}")
