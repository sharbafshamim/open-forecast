"""Forecast combination: equal-weight average of Chronos-2 and LEAR (both with exogenous inputs).

Usage:  python analysis/price_ensemble.py results/price_<market> [published_forecasts.csv]

Reads the committed per-day forecasts of a price run (price_predictions.csv[.gz]),
averages the two members hour by hour, and writes into the same folder:

  ensemble_metrics.csv   MAE, RMSE, rMAE of the ensemble and of its members
  ensemble_dm.csv        one-sided multivariate DM tests of the ensemble against
                         every other forecast (i.i.d. and Newey-West variance),
                         including the published Lago et al. (2021) ensembles
                         when their file is given

The combination uses no extra information and no fitted weights, so it stays
inside the members' information set and needs no extra calibration data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_forecast.significance import holm, multivariate_dm  # noqa: E402

MEMBERS = ("chronos2|price_exogenous", "lear|price_exogenous")
NAME = "ensemble: Chronos-2 + LEAR (equal weights)"


def read_predictions(folder: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    path = next((folder / n for n in ("price_predictions.csv.gz", "price_predictions.csv")
                 if (folder / n).exists()), None)
    if path is None:
        raise FileNotFoundError(f"no price_predictions.csv[.gz] in {folder}")
    p = pd.read_csv(path, parse_dates=["day"])
    p["hour"] = p["hour"].astype(str).str.lstrip("h").astype(int)
    p["key"] = p["model"] + "|" + p["information_set"].fillna("-")
    wide = {k: g.pivot(index="day", columns="hour", values="predicted").sort_index(axis=1)
            for k, g in p.groupby("key")}
    truth = p.drop_duplicates(["day", "hour"]).pivot(index="day", columns="hour", values="actual").sort_index(axis=1)
    return wide, truth


def published(path: Path) -> dict[str, pd.DataFrame]:
    pub = pd.read_csv(path, index_col=0, parse_dates=True)
    out = {}
    for column in ("LEAR Ensemble", "DNN Ensemble"):
        frame = pub[column].to_frame("v")
        frame["day"], frame["hour"] = frame.index.normalize(), frame.index.hour
        out[f"{column.split()[0]} ensemble (published)"] = frame.pivot(index="day", columns="hour", values="v")
    return out


def run(folder: Path, published_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide, truth = read_predictions(folder)
    missing = [m for m in MEMBERS if m not in wide]
    if missing:
        raise ValueError(f"{folder}: members missing from the forecasts: {missing}")
    if published_path is not None and Path(published_path).exists():
        wide.update(published(Path(published_path)))
    days = truth.index
    for frame in wide.values():
        days = days.intersection(frame.index)
    wide = {k: v.loc[days] for k, v in wide.items()}
    t = truth.loc[days].to_numpy()
    ensemble = (wide[MEMBERS[0]].to_numpy() + wide[MEMBERS[1]].to_numpy()) / 2
    naive_key = next(k for k in wide if k.startswith("naive|"))
    naive_mae = np.abs(wide[naive_key].to_numpy() - t).mean()

    def scores(p: np.ndarray) -> dict:
        e = p - t
        return {"mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()), "rmae": np.abs(e).mean() / naive_mae}

    metrics = pd.DataFrame([{"forecast": NAME, "days": len(days), **scores(ensemble)}] +
                           [{"forecast": k, "days": len(days), **scores(v.to_numpy())}
                            for k, v in wide.items()]).sort_values("rmae").reset_index(drop=True)
    rows = []
    for k, v in wide.items():
        other = v.to_numpy()
        # H1 "ensemble more accurate than k", and the reverse direction
        rows.append({"versus": k, "p_iid": multivariate_dm(t, other, ensemble)[1],
                     "p_newey_west": multivariate_dm(t, other, ensemble, lags=7)[1],
                     "p_other_better_newey_west": multivariate_dm(t, ensemble, other, lags=7)[1]})
    dm = pd.DataFrame(rows)
    dm["p_holm_newey_west"] = holm(dm["p_newey_west"])
    metrics.round(4).to_csv(folder / "ensemble_metrics.csv", index=False)
    dm.round(6).to_csv(folder / "ensemble_dm.csv", index=False)
    return metrics, dm


if __name__ == "__main__":
    m, d = run(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
    pd.set_option("display.width", 200)
    print(m.round(4).to_string(index=False))
    print(d.round(5).to_string(index=False))
