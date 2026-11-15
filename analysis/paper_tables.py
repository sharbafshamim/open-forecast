"""Build every table, test and result figure of the paper from results/.

Usage:  python analysis/paper_tables.py [results_dir]

Reads the folders written by scripts/reproduce.sh and writes results/paper/:

  load_isone_rolling.csv        Table 3: ISO-NE load, 13 monthly-refit windows of 2025
  load_isone_forecast_weather.csv   ISO-NE load with archived weather forecasts at test time
  dm_isone_load.csv             per-zone DM tests between the main load models (Holm-adjusted)
  price_isone.csv               Table 4: ISO-NE price, incl. peak-hour and spike-hour MAE
  price_epf_rmae.csv            Table 6: EPF markets (rMAE), published forecasts appended
  tetouan_single_split.csv      Tetouan conventional test window
  tetouan_rolling.csv           Table 5: Tetouan 11 rolling windows (+ window tests vs naive)
  figure_isone_rolling.png/pdf  per-window RMSE on ISO-NE
  figure_tetouan_rolling.png/pdf  per-window RMSE on Tetouan
Missing inputs are skipped, so the script can be run while results arrive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_forecast.significance import daily_dm_test, dm_test, holm, window_test  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = ROOT / "paper"
OUT.mkdir(parents=True, exist_ok=True)
SET = {"weather_time": "A", "weather_time_history": "B", "weather_time_history_weekly": "B+"}
H = 24
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#52514e", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def show(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}\n{frame.to_string()}")


# --------------------------------------------------------------------------
# Load: pooled metrics from the committed test predictions of rolling runs
# --------------------------------------------------------------------------
def rolling_predictions(folders: list[str]) -> pd.DataFrame:
    frames = []
    for folder in folders:
        path = ROOT / folder / "rolling_test_predictions.csv.gz"
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["forecast_origin"]))
    if not frames:
        return pd.DataFrame()
    pred = pd.concat(frames, ignore_index=True)
    return pred.drop_duplicates(["model", "feature_mode", "forecast_origin", "zone"])


def load_metrics(pred: pd.DataFrame, peak_share: float = 0.10) -> pd.DataFrame:
    """Aggregate RMSE/nRMSE (pooled), mean zonal nRMSE, MAPE and peak-hour RMSE.

    Peak hours: the target hours whose actual system load (sum of zones) is in
    the top ``peak_share`` of the evaluation period.
    """
    system = pred.groupby(["model", "feature_mode", "forecast_origin"])["actual"].sum()
    any_key = system.index[0][:2]
    threshold = system.loc[any_key].quantile(1 - peak_share)
    peak_origins = set(system.loc[any_key][system.loc[any_key] >= threshold].index)
    rows = []
    for (model, mode), g in pred.groupby(["model", "feature_mode"]):
        e = g["actual"] - g["predicted"]
        zonal = g.assign(se=e ** 2).groupby("zone").apply(
            lambda z: np.sqrt(z["se"].mean()) / z["actual"].mean() * 100, include_groups=False)
        peak = g[g["forecast_origin"].isin(peak_origins)]
        ep = peak["actual"] - peak["predicted"]
        rows.append({"model": model, "set": SET[mode], "origins": g["forecast_origin"].nunique(),
                     "rmse": np.sqrt((e ** 2).mean()), "mae": e.abs().mean(),
                     "nrmse_pct": np.sqrt((e ** 2).mean()) / g["actual"].mean() * 100,
                     "mean_zonal_nrmse_pct": zonal.mean(),
                     "mape_pct": (e.abs() / g["actual"].abs()).mean() * 100,
                     "peak_hour_rmse": np.sqrt((ep ** 2).mean())})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def window_ranks(folders: list[str], keep_from: int = 0) -> pd.DataFrame:
    frames = [pd.read_csv(ROOT / f / "rolling_metrics.csv") for f in folders
              if (ROOT / f / "rolling_metrics.csv").exists()]
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames)
    a = m[(m.scope == "aggregate") & (m.fold >= keep_from)].drop_duplicates(["fold", "model", "feature_mode"]).copy()
    a["rank"] = a.groupby("fold")["rmse"].rank()
    return a


def rank_summary(a: pd.DataFrame, benchmark=("seasonal_naive", "weather_time_history_weekly")) -> pd.DataFrame:
    bench = a[(a.model == benchmark[0]) & (a.feature_mode == benchmark[1])].set_index("fold")["rmse"]
    rows = []
    for (model, mode), g in a.groupby(["model", "feature_mode"]):
        g = g.set_index("fold")
        row = {"model": model, "set": SET[mode], "windows": len(g), "mean_rank": g["rank"].mean(),
               "wins": int((g["rank"] == 1).sum()), "mean_rmse": g["rmse"].mean()}
        if len(bench) and (model, mode) != benchmark:
            common = g.index.intersection(bench.index)
            beats, p = window_test(g.loc[common, "rmse"], bench.loc[common])
            row.update({"beats_naive_windows": beats, "p_wilcoxon_vs_naive": p})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)


def dm_pairs(pred: pd.DataFrame, pairs: list[tuple[tuple[str, str], tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for (ma, sa), (mb, sb) in pairs:
        a = pred[(pred.model == ma) & (pred.feature_mode == sa)].set_index(["forecast_origin", "zone"])
        b = pred[(pred.model == mb) & (pred.feature_mode == sb)].set_index(["forecast_origin", "zone"])
        if a.empty or b.empty:
            continue
        for zone in sorted(a.index.get_level_values("zone").unique()):
            x = a.xs(zone, level="zone")
            y = b.xs(zone, level="zone").reindex(x.index)
            ea, eb = (x.actual - x.predicted).to_numpy(), (y.actual - y.predicted).to_numpy()
            stat, p = dm_test(ea, eb, H, lags=168)
            _, pd_ = daily_dm_test(ea, eb, (x.index + pd.Timedelta(hours=H)).normalize().to_numpy())
            rows.append({"model": f"{ma} ({SET[sa]})", "versus": f"{mb} ({SET[sb]})", "zone": zone,
                         "rmse_ratio": np.sqrt((ea ** 2).mean() / (eb ** 2).mean()), "dm": stat,
                         "p": p, "p_daily": pd_})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"])
        out["p_daily_holm"] = holm(out["p_daily"])
    return out


def isone_load() -> None:
    folders = ["isone_rolling_trees", "isone_rolling_foundation", "isone_rolling_lstm",
               "isone_rolling_cross_attention_transformer", "isone_rolling_patchtst"]
    pred = rolling_predictions(folders)
    if pred.empty:
        return
    table = load_metrics(pred)
    ranks = rank_summary(window_ranks(folders))
    table = table.merge(ranks[["model", "set", "mean_rank", "wins"]], on=["model", "set"], how="left")
    table.round(3).to_csv(OUT / "load_isone_rolling.csv", index=False)
    show("ISO-NE load, 13 monthly-refit windows (MW)", table.round(2))

    B, BP = "weather_time_history", "weather_time_history_weekly"
    best = table.iloc[0]
    best_key = (best.model, {v: k for k, v in SET.items()}[best.set])
    others = [(m, s) for m, s in [("hist_gradient_boosting", B), ("hist_gradient_boosting", BP), ("lstm", B),
                                   ("lstm", BP), ("chronos2", BP), ("persistence", B)] if (m, s) != best_key]
    pairs = [(best_key, o) for o in others] + [(("chronos2", BP), ("hist_gradient_boosting", B))]
    dm = dm_pairs(pred, pairs)
    if len(dm):
        dm.round(5).to_csv(OUT / "dm_isone_load.csv", index=False)
        summary = dm.assign(better=(dm.dm < 0) & (dm.p_holm < 0.05), worse=(dm.dm > 0) & (dm.p_holm < 0.05),
                            better_daily=(dm.dm < 0) & (dm.p_daily_holm < 0.05)) \
                    .groupby(["model", "versus"])[["better", "worse", "better_daily"]].sum()
        show("Zones (of 8) significantly better / worse, Holm-adjusted (hourly NW-168 and daily tests)", summary)
        summary.to_csv(OUT / "dm_isone_load_summary.csv")

    fw = rolling_predictions(["isone_rolling_forecastweather_trees", "isone_rolling_forecastweather_lstm"])
    if not fw.empty:
        t = load_metrics(fw)
        obs = table.set_index(["model", "set"])["rmse"]
        t["rmse_observed_weather"] = [obs.get((m, s), np.nan) for m, s in zip(t.model, t.set)]
        t.round(3).to_csv(OUT / "load_isone_forecast_weather.csv", index=False)
        show("ISO-NE load with archived day-ahead weather forecasts at test time (MW)", t.round(2))

    a = window_ranks(folders)
    keep = [("hist_gradient_boosting", B, "Gradient boosting (B)"), ("lstm", BP, "LSTM (B+)"),
            ("lstm", B, "LSTM (B)"), ("chronos2", BP, "Chronos-2 (zero-shot)"),
            ("random_forest", BP, "Random forest (B+)"), ("persistence", B, "Persistence")]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for i, (model, mode, label) in enumerate(keep):
        s = a[(a.model == model) & (a.feature_mode == mode)].sort_values("fold")
        if len(s):
            ax.plot(pd.to_datetime(s.test_start), s.rmse, marker="o", ms=3.5, color=SERIES[i], label=label,
                    ls=":" if model == "persistence" else "-")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Aggregate RMSE (MW), 24 h ahead")
    ax.set_xlabel("Start of 28-day test window (2025); models refit before each window")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(OUT / "figure_isone_rolling.png")
    fig.savefig(OUT / "figure_isone_rolling.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Price
# --------------------------------------------------------------------------
def read_price_predictions(folder: Path) -> pd.DataFrame:
    for name in ("price_predictions.csv.gz", "price_predictions.csv"):
        if (folder / name).exists():
            p = pd.read_csv(folder / name, parse_dates=["day"])
            p["hour"] = p["hour"].astype(str).str.lstrip("h").astype(int)
            return p
    return pd.DataFrame()


def price_tables() -> None:
    folder = ROOT / "price_isone_price"
    if (folder / "price_metrics.csv").exists():
        m = pd.read_csv(folder / "price_metrics.csv")
        pred = read_price_predictions(folder)
        if not pred.empty:
            spike = pred.drop_duplicates(["day", "hour"])["actual"].quantile(0.95)
            extra = []
            for (model, info), g in pred.groupby(["model", "information_set"]):
                e = (g.actual - g.predicted).abs()
                peak = g.hour.between(7, 22)          # hour-beginning 07-22 (on-peak block)
                extra.append({"model": model, "information_set": info if model != "naive" else "-",
                              "peak_mae": e[peak].mean(), "spike_mae": e[g.actual >= spike].mean()})
            m = m.merge(pd.DataFrame(extra), on=["model", "information_set"], how="left")
        m.round(3).to_csv(OUT / "price_isone.csv", index=False)
        show("ISO-NE price, test year 2025", m.round(3))
    ensembles = []
    for market, folder in [("ISO-NE", ROOT / "price_isone_price")] + \
            [(mk.upper(), ROOT / f"price_epf_{mk}") for mk in ["pjm", "np", "de", "be", "fr"]]:
        if (folder / "ensemble_metrics.csv").exists():
            e = pd.read_csv(folder / "ensemble_metrics.csv").assign(market=market)
            d = pd.read_csv(folder / "ensemble_dm.csv").assign(market=market)
            ensembles.append((e, d))
    if ensembles:
        e = pd.concat([x for x, _ in ensembles])
        d = pd.concat([y for _, y in ensembles])
        wide = e.pivot_table(index="forecast", columns="market", values="rmae")
        wide = wide[[c for c in ["ISO-NE", "PJM", "NP", "DE", "BE", "FR"] if c in wide.columns]]
        wide.round(3).to_csv(OUT / "price_ensemble_rmae.csv")
        d.round(6).to_csv(OUT / "price_ensemble_dm.csv", index=False)
        show("Forecast combination Chronos-2 + LEAR: rMAE by market", wide.round(3))
        show("Ensemble vs each forecast (one-sided multivariate DM)", d.round(4))

    frames = []
    for market in ["pjm", "np", "de", "be", "fr"]:
        path = ROOT / f"price_epf_{market}" / "price_metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path).assign(market=market.upper()))
    if frames:
        wide = pd.concat(frames).pivot_table(index=["model", "information_set"], columns="market", values="rmae")
        wide = wide[[c for c in ["PJM", "NP", "DE", "BE", "FR"] if c in wide.columns]]
        published = ROOT / "epf_published_comparison" / "comparison.csv"
        if published.exists():
            pub = pd.read_csv(published)
            pub = pub[pub.model.str.contains("published")].pivot(index="model", columns="market", values="rMAE")
            pub.index = pd.MultiIndex.from_tuples([(i, "published (Lago et al. 2021)") for i in pub.index])
            wide = pd.concat([wide, pub[wide.columns]])
        wide.round(3).to_csv(OUT / "price_epf_rmae.csv")
        show("EPF markets, rMAE", wide.round(3))


# --------------------------------------------------------------------------
# Tetouan
# --------------------------------------------------------------------------
def tetouan() -> None:
    rows = []
    for folder in ["tetouan_v2_trees", "tetouan_v2_deep", "tetouan_v2_foundation"]:
        path = ROOT / folder / "metrics.csv"
        if path.exists():
            m = pd.read_csv(path)
            m = m[(m.split == "test") & (m.scope == "aggregate")]
            rows.append(m.assign(set=m.feature_mode.map(SET))[["model", "set", "rmse", "nrmse_pct"]])
    if rows:
        t = pd.concat(rows).drop_duplicates(["model", "set"]).sort_values("rmse")
        t.round(2).to_csv(OUT / "tetouan_single_split.csv", index=False)
        show("Tetouan, conventional test window (dataset units)", t.round(2).reset_index(drop=True))

    folders = ["tetouan_rolling_trees", "tetouan_rolling_deep", "tetouan_rolling_foundation"]
    a = window_ranks(folders)
    if a.empty:
        return
    full = rank_summary(a)
    late = rank_summary(window_ranks(folders, keep_from=2))     # >= ~3 months of training data
    full = full.merge(late[["model", "set", "mean_rank", "wins"]].rename(
        columns={"mean_rank": "mean_rank_from_window3", "wins": "wins_from_window3"}), on=["model", "set"], how="left")
    full.round(3).to_csv(OUT / "tetouan_rolling.csv", index=False)
    show("Tetouan rolling origin, 11 windows", full.round(2))

    keep = [("chronos_bolt_small", "weather_time_history_weekly", "Chronos-Bolt small (zero-shot)", "-"),
            ("chronos2", "weather_time_history_weekly", "Chronos-2 (zero-shot)", "-"),
            ("random_forest", "weather_time_history_weekly", "Random forest (B+)", "-"),
            ("extra_trees", "weather_time_history_weekly", "Extra trees (B+)", "-"),
            ("lstm", "weather_time_history_weekly", "LSTM (B+)", "-"),
            ("seasonal_naive", "weather_time_history_weekly", "Weekly seasonal naive", "--"),
            ("persistence", "weather_time_history", "Persistence", ":")]
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    for i, (model, mode, label, style) in enumerate(keep):
        s = a[(a.model == model) & (a.feature_mode == mode)].sort_values("fold")
        if len(s):
            ax.plot(pd.to_datetime(s.test_start), s.rmse, marker="o", ms=3.5, ls=style, color=SERIES[i], label=label)
    top = ax.get_ylim()[1]
    for start, end, text in (("2017-05-27", "2017-06-27", "Ramadan / Eid al-Fitr"),
                             ("2017-08-31", "2017-09-03", "Eid al-Adha")):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=GRID, alpha=0.8, lw=0, zorder=0)
        ax.text(pd.Timestamp(start), top, text, fontsize=7, color=MUTED, va="top")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Aggregate RMSE (dataset units)")
    ax.set_xlabel("Start of 28-day test window (2017)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(OUT / "figure_tetouan_rolling.png")
    fig.savefig(OUT / "figure_tetouan_rolling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    isone_load()
    price_tables()
    tetouan()
    print(f"\nWrote {OUT}")
