"""Build the tables and the rolling-origin figure used in the paper.

Usage:  python analysis/paper_tables.py [results_dir]

Reads the committed result folders and writes results/paper/:
  table_load_aggregate_rmse.csv   Table 3 (single seed; deep B/B+ five-seed means in seed_summary.csv)
  table_price_isone.csv           Table 4
  table_tetouan_rolling.csv       Table 5
  table_price_epf_rmae.csv        Table 6
  figure_tetouan_rolling.png/pdf  Figure 6
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = ROOT / "paper"
OUT.mkdir(parents=True, exist_ok=True)
SET = {"weather_time": "A", "weather_time_history": "B", "weather_time_history_weekly": "B+"}
DEEP = {"lstm", "cross_attention_transformer", "patchtst"}


def load_table() -> None:
    rows = []
    for dataset, folders in {"ISO-NE": ["isone_trees", "isone_foundation"],
                             "Tetouan": ["tetouan_v2_trees", "tetouan_v2_foundation"]}.items():
        for folder in folders:
            path = ROOT / folder / "metrics.csv"
            if path.exists():
                m = pd.read_csv(path)
                m = m[(m.split == "test") & (m.scope == "aggregate")]
                rows.append(pd.DataFrame({"dataset": dataset, "model": m.model,
                                          "set": m.feature_mode.map(SET), "rmse": m.rmse,
                                          "nrmse_pct": m.nrmse_pct}))
    deep = ROOT / "deep" / "deep_metrics.csv"
    if deep.exists():
        d = pd.read_csv(deep)
        d = d[d.scope == "aggregate"]
        rows.append(pd.DataFrame({"dataset": d.dataset.map({"isone": "ISO-NE", "tetouan": "Tetouan"}),
                                  "model": d.model, "set": d.set, "rmse": d.RMSE, "nrmse_pct": d["nRMSE_%"]}))
    table = pd.concat(rows).drop_duplicates(["dataset", "model", "set"])
    wide = table.pivot_table(index=["dataset", "model"], columns="set", values="rmse").round(1)
    wide["nRMSE_B_pct"] = table[table.set == "B"].set_index(["dataset", "model"]).nrmse_pct.round(2)
    wide.to_csv(OUT / "table_load_aggregate_rmse.csv")
    print("\nLoad, aggregate test RMSE (ISO-NE: MW; Tetouan: dataset units)\n", wide.to_string())


def price_tables() -> None:
    isone = ROOT / "price_isone_price" / "price_metrics.csv"
    if isone.exists():
        m = pd.read_csv(isone)[["model", "information_set", "mae", "rmse", "smape_pct", "rmae"]].round(3)
        m.to_csv(OUT / "table_price_isone.csv", index=False)
        print("\nISO-NE price, test year 2025\n", m.to_string(index=False))
    frames = []
    for market in ["pjm", "np", "de", "be", "fr"]:
        path = ROOT / f"price_epf_{market}" / "price_metrics.csv"
        if path.exists():
            m = pd.read_csv(path)
            m["market"] = market.upper()
            frames.append(m)
    if frames:
        wide = pd.concat(frames).pivot_table(index=["model", "information_set"], columns="market",
                                             values="rmae")[["PJM", "NP", "DE", "BE", "FR"]].round(3)
        wide.to_csv(OUT / "table_price_epf_rmae.csv")
        print("\nEPF markets, rMAE\n", wide.to_string())


def rolling() -> None:
    paths = [ROOT / f / "rolling_metrics.csv" for f in ("tetouan_rolling_trees", "tetouan_rolling_deep_foundation")]
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        return
    m = pd.concat(frames)
    a = m[m.scope == "aggregate"].drop_duplicates(["fold", "model", "feature_mode"]).copy()
    a = a[~(a.model.isin(DEEP) & (a.feature_mode == "weather_time_history_weekly"))]
    a["rank"] = a.groupby("fold").rmse.rank()
    summary = (a.groupby(["model", "feature_mode"])
                .agg(windows=("fold", "nunique"), mean_rank=("rank", "mean"),
                     wins=("rank", lambda r: int((r == 1).sum())), mean_rmse=("rmse", "mean"))
                .sort_values("mean_rank").round(2))
    summary.to_csv(OUT / "table_tetouan_rolling.csv")
    print("\nTetouan rolling origin, 11 x 28-day windows\n", summary.to_string())

    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#52514e"]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": grid,
                         "grid.linewidth": 0.6, "lines.linewidth": 1.8, "savefig.dpi": 300,
                         "savefig.bbox": "tight"})
    keep = [("chronos_bolt_small", "weather_time_history", "Chronos-Bolt small (zero-shot)", "-"),
            ("chronos2", "weather_time_history", "Chronos-2 (zero-shot)", "-"),
            ("random_forest", "weather_time_history_weekly", "Random forest (B+)", "-"),
            ("extra_trees", "weather_time_history_weekly", "Extra trees (B+)", "-"),
            ("lstm", "weather_time_history", "LSTM (B)", "-"),
            ("seasonal_naive", "weather_time_history", "Weekly seasonal naive", "--"),
            ("persistence", "weather_time_history", "Persistence", ":")]
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    for i, (model, mode, label, style) in enumerate(keep):
        s = a[(a.model == model) & (a.feature_mode == mode)].sort_values("fold")
        if len(s):
            ax.plot(pd.to_datetime(s.test_start), s.rmse, marker="o", ms=3.5, ls=style,
                    color=series[i], label=label)
    top = ax.get_ylim()[1]
    for start, end, text in (("2017-05-27", "2017-06-27", "Ramadan / Eid al-Fitr"),
                             ("2017-08-31", "2017-09-03", "Eid al-Adha")):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=grid, alpha=0.8, lw=0, zorder=0)
        ax.text(pd.Timestamp(start), top, text, fontsize=7, color=muted, va="top")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Aggregate RMSE (dataset units)")
    ax.set_xlabel("Start of 28-day test window (2017)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(OUT / "figure_tetouan_rolling.png")
    fig.savefig(OUT / "figure_tetouan_rolling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    load_table()
    price_tables()
    rolling()
    print(f"\nWrote {OUT}")
