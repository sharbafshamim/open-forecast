"""Diebold-Mariano tests of each model against a benchmark, per zone.

Usage: python analysis/dm_tests.py results/<run>/predictions.csv[.gz] <benchmark> [horizon]

Squared-error loss, HLN correction, Newey-West variance with a one-week
bandwidth (168 hourly origins). Also reports the test on daily-averaged loss
differentials and Holm-adjusted p-values over all model x zone comparisons.
A negative statistic means the model has smaller squared error than the benchmark.
"""
import re
import sys

import numpy as np
import pandas as pd

from open_forecast.significance import daily_dm_test, dm_test, holm


def run(path, benchmark, h=24):
    pred = pd.read_csv(path, parse_dates=["forecast_origin"])
    actual_col = next(c for c in pred.columns if c.startswith("actual"))
    pred_col = next(c for c in pred.columns if c.startswith("predicted"))
    bench = pred[pred.model == benchmark].set_index(["forecast_origin", "zone"])
    rows = []
    for (model, mode), g in pred[pred.model != benchmark].groupby(["model", "feature_mode"]):
        g = g.set_index(["forecast_origin", "zone"])
        for zone in sorted(g.index.get_level_values("zone").unique()):
            a = g.xs(zone, level="zone")
            b = bench.xs(zone, level="zone").reindex(a.index)
            e_m = (a[actual_col] - a[pred_col]).to_numpy()
            e_b = (b[actual_col] - b[pred_col]).to_numpy()
            stat, p = dm_test(e_m, e_b, h)
            target_day = (a.index + pd.Timedelta(hours=h)).normalize()
            _, p_daily = daily_dm_test(e_m, e_b, target_day.to_numpy())
            rows.append({"model": model, "feature_mode": mode, "zone": zone,
                         "rmse_ratio_vs_benchmark": np.sqrt((e_m ** 2).mean()) / np.sqrt((e_b ** 2).mean()),
                         "dm_stat": stat, "p_value": p, "p_value_daily": p_daily, "n": len(e_m)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"])
    return out


if __name__ == "__main__":
    out = run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv(re.sub(r"predictions\.csv(\.gz)?$", f"dm_vs_{sys.argv[2]}.csv", sys.argv[1]), index=False)
