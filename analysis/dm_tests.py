"""Diebold-Mariano tests (squared-error loss, HLN small-sample correction,
Newey-West variance with h-1 lags) of each model against a benchmark, per zone.

Usage: python analysis/dm_tests.py results/<run>/predictions.csv <benchmark> [horizon]
Positive DM statistic => the model has larger squared error than the benchmark.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats


def dm_test(e_model, e_bench, h):
    d = e_model ** 2 - e_bench ** 2
    n = len(d)
    d_bar = d.mean()
    dc = d - d_bar
    gamma = [np.dot(dc[k:], dc[: n - k]) / n for k in range(h)]
    var = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))  # Bartlett weights
    var = max(var, 1e-12)
    dm = d_bar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p


def run(path, benchmark, h=24):
    pred = pd.read_csv(path, parse_dates=["forecast_origin"])
    value_cols = [c for c in pred.columns if c.startswith("actual")] + [c for c in pred.columns if c.startswith("predicted")]
    actual_col, pred_col = value_cols[0], value_cols[1]
    bench = pred[pred.model == benchmark].set_index(["forecast_origin", "zone"])
    rows = []
    for (model, mode), g in pred[pred.model != benchmark].groupby(["model", "feature_mode"]):
        if model in ("persistence", "seasonal_naive") and model != benchmark and mode != "weather_time_history":
            continue
        g = g.set_index(["forecast_origin", "zone"])
        for zone in sorted(g.index.get_level_values("zone").unique()):
            a = g.xs(zone, level="zone")
            b = bench.xs(zone, level="zone").reindex(a.index)
            e_m = (a[actual_col] - a[pred_col]).to_numpy()
            e_b = (b[actual_col] - b[pred_col]).to_numpy()
            stat, p = dm_test(e_m, e_b, h)
            rows.append({"model": model, "feature_mode": mode, "zone": zone,
                         "rmse_ratio_vs_benchmark": np.sqrt((e_m ** 2).mean()) / np.sqrt((e_b ** 2).mean()),
                         "dm_stat": stat, "p_value": p, "n": len(e_m)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv(sys.argv[1].replace("predictions.csv", f"dm_vs_{sys.argv[2]}.csv"), index=False)
