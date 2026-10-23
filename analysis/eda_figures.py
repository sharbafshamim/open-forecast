"""Exploratory data figures for the paper.

    python analysis/eda_figures.py [--data data/raw] [--out results/eda]

ISO New England (main dataset)
  isone_fig1_overview        daily system load and day-ahead price, 2023-2025 (stacked panels)
  isone_fig2_load_temperature daily load vs temperature: heating / cooling response
  isone_fig3_daily_profiles  mean hourly load profile by season, weekday vs weekend
  isone_fig4_price_tail      hourly DA price distribution (log-count) and share of extreme hours
  isone_fig5_zone_sizes      mean and peak load per load zone
Tetouan (secondary dataset)
  tetouan_fig1_week          a regular (non-holiday) week, Zone 1 and Zone 2
  tetouan_fig2_daily_profile mean hourly profile per zone
  tetouan_fig3_monthly       monthly mean consumption per zone
  tetouan_fig4_correlation   correlation of targets and weather inputs

All figures use the same loaders (and daylight-saving policies) as the experiments,
and a manifest records the SHA-256 of every raw file used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from open_forecast.isone import ZONES, load_isone_smd  # noqa: E402
from open_forecast.preprocessing import load_hourly_dataset  # noqa: E402
from open_forecast.price import load_isone_price  # noqa: E402

# Same reference palette and styling as analysis/summarize.py
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL_BLUE = "Blues"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
TETOUAN_WEATHER = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows"]


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# ISO-NE
# --------------------------------------------------------------------------
def isone_figures(paths: list[Path], out: Path) -> None:
    zones, _ = load_isone_smd(paths, ZONES)
    system = load_isone_price(paths).hourly           # Price, RT_Demand, Dry_Bulb, Dew_Point
    daily = system.resample("D").agg({"RT_Demand": "mean", "Price": "mean", "Dry_Bulb": "mean"})

    # Fig 1: two stacked panels sharing the time axis (never a dual y-axis)
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
    a.plot(daily.index, daily["RT_Demand"] / 1000, color=SERIES[0])
    a.set_ylabel("System load (GW)\ndaily mean")
    b.plot(daily.index, daily["Price"], color=SERIES[1])
    b.set_ylabel("DA LMP ($/MWh)\ndaily mean")
    for ax in (a, b):
        ax.axvspan(pd.Timestamp("2025-01-01"), daily.index.max(), color=GRID, alpha=0.6, lw=0, zorder=0)
    a.text(pd.Timestamp("2025-01-08"), a.get_ylim()[1], "test year", va="top", fontsize=8, color=MUTED)
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    save(fig, out, "isone_fig1_overview")

    # Fig 2: load-temperature response, coloured by year (3 series)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for i, year in enumerate(sorted(daily.index.year.unique())):
        d = daily[daily.index.year == year]
        ax.scatter(d["Dry_Bulb"], d["RT_Demand"] / 1000, s=9, color=SERIES[i], alpha=0.7,
                   label=str(year), linewidths=0)
    ax.set_xlabel("Daily mean dry-bulb temperature (°F)")
    ax.set_ylabel("Daily mean system load (GW)")
    ax.legend(loc="upper center", ncol=3)
    save(fig, out, "isone_fig2_load_temperature")

    # Fig 3: mean daily profile by season (small multiples: weekday vs weekend)
    frame = system[["RT_Demand"]].copy()
    frame["season"] = frame.index.month.map(SEASONS)
    frame["hour"] = frame.index.hour
    frame["daytype"] = np.where(frame.index.dayofweek >= 5, "Weekend", "Weekday")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, daytype in zip(axes, ("Weekday", "Weekend")):
        sub = frame[frame["daytype"] == daytype]
        for i, season in enumerate(("Winter", "Spring", "Summer", "Autumn")):
            profile = sub[sub["season"] == season].groupby("hour")["RT_Demand"].mean() / 1000
            ax.plot(profile.index, profile.values, color=SERIES[i], label=season)
        ax.set_title(daytype, fontsize=9, color=INK)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean system load (GW)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    save(fig, out, "isone_fig3_daily_profiles")

    # Fig 4: price distribution, log-count to show the tail
    price = system["Price"]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.hist(price, bins=120, color=SERIES[1], log=True)
    p99 = price.quantile(0.99)
    ax.axvline(p99, color=MUTED, lw=1, ls="--")
    ax.text(p99, ax.get_ylim()[1], f" 99th pct = ${p99:.0f}", va="top", fontsize=8, color=MUTED)
    ax.set_xlabel("Hourly DA LMP ($/MWh)")
    ax.set_ylabel("Hours (log scale)")
    stats = {"mean": price.mean(), "median": price.median(), "p99": p99, "max": price.max(),
             "share_hours_above_100": float((price > 100).mean())}
    save(fig, out, "isone_fig4_price_tail")

    # Fig 5: zone sizes (one measure per chart; mean and peak as two panels)
    mean = zones[ZONES].mean().sort_values()
    peak = zones[ZONES].max().reindex(mean.index)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, values, title in ((axes[0], mean, "Mean load (MW)"), (axes[1], peak, "Peak load (MW)")):
        ax.barh(values.index, values.values, color=SERIES[0], height=0.6)
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
        for y, v in enumerate(values.values):
            ax.text(v, y, f" {v:,.0f}", va="center", fontsize=7, color=MUTED)
    save(fig, out, "isone_fig5_zone_sizes")

    summary = pd.DataFrame({"mean_MW": zones[ZONES].mean(), "peak_MW": zones[ZONES].max(),
                            "min_MW": zones[ZONES].min()}).round(1)
    summary.to_csv(out / "isone_zone_summary.csv")
    (out / "isone_price_summary.json").write_text(json.dumps({k: round(float(v), 4) for k, v in stats.items()}, indent=2))


# --------------------------------------------------------------------------
# Tetouan
# --------------------------------------------------------------------------
def tetouan_figures(path: Path, out: Path) -> None:
    hourly, _ = load_hourly_dataset(path, "DateTime", ["Zone 1", "Zone 2"], TETOUAN_WEATHER)
    targets = ["Zone 1", "Zone 2"]

    # The source does not document units; values are shown as published (x 1000).
    unit = "Consumption (x1000, dataset units)"
    # A regular week (no holiday); 26 Jun 2017, used in v1, was Eid al-Fitr.
    week = hourly.loc["2017-10-09":"2017-10-15 23:00", targets] / 1000
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, zone in enumerate(targets):
        ax.plot(week.index, week[zone], color=SERIES[i], label=zone)
    ax.set_ylabel(unit)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "tetouan_fig1_week")

    fig, ax = plt.subplots(figsize=(4.8, 3))
    for i, zone in enumerate(targets):
        profile = hourly[zone].groupby(hourly.index.hour).mean() / 1000
        ax.plot(profile.index, profile.values, color=SERIES[i], label=zone, marker="o", ms=3)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(unit.replace("Consumption", "Mean consumption"))
    ax.legend()
    save(fig, out, "tetouan_fig2_daily_profile")

    monthly = hourly[targets].resample("MS").mean() / 1000
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(monthly))
    for i, zone in enumerate(targets):
        ax.bar(x + (i - 0.5) * 0.38, monthly[zone], width=0.36, color=SERIES[i], label=zone)
    ax.set_xticks(x, monthly.index.strftime("%b"))
    ax.set_ylabel(unit.replace("Consumption", "Monthly mean"))
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2)
    save(fig, out, "tetouan_fig3_monthly")

    corr = hourly[targets + TETOUAN_WEATHER].corr()
    fig, ax = plt.subplots(figsize=(4.8, 4))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)   # diverging, neutral midpoint at 0
    labels = [c.replace("general diffuse flows", "GHI").replace("diffuse flows", "DHI") for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for (r, c), v in np.ndenumerate(corr.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    save(fig, out, "tetouan_fig4_correlation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/eda"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    isone = [args.data / "isone" / f"{y}_smd_hourly.xlsx" for y in (2023, 2024, 2025)]
    if all(p.exists() for p in isone):
        isone_figures(isone, args.out)
        manifest.update({p.name: sha256(p) for p in isone})
    else:
        print("ISO-NE workbooks not found; skipping (see data/README.md)")

    tetouan = args.data / "power_consumption.csv"
    if tetouan.exists():
        tetouan_figures(tetouan, args.out)
        manifest[tetouan.name] = sha256(tetouan)
    else:
        print("Tetouan CSV not found; skipping (see data/README.md)")

    (args.out / "eda_manifest.json").write_text(json.dumps({"raw_data_sha256": manifest}, indent=2))


if __name__ == "__main__":
    main()
