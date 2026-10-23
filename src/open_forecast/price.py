"""Day-ahead electricity price forecasting under a declared information set.

Protocol (the standard EPF setting, Lago et al. 2021, Applied Energy 293:116983)
------------------------------------------------------------------------------
* On day D-1 a forecaster submits all 24 hourly prices of delivery day D.
* Prices of day D-1 and earlier are known (the D-1 auction cleared on D-2).
* Exogenous inputs for day D are allowed only when they are *forecasts*
  published before the auction (EPF datasets: day-ahead load / generation
  forecasts).  ISO-NE SMD files contain no such forecast, so observed day-D
  weather is offered as an explicitly declared proxy, exactly as in the load
  experiments, and every run reports a price-only information set as well.
* Market-cleared day-D quantities (ISO-NE ``DA_Demand``) are never used: they
  are produced by the same auction as the price being forecast.
* Models are recalibrated on a rolling (or expanding) calibration window every
  ``recalibration_days`` days, using only data strictly before day D.

Information sets
----------------
``price_only``      prices D-1, D-2, D-3, D-7 (24 h each) + day-of-week dummies
``price_exogenous`` price_only + exogenous series at D, D-1 and D-7
                    (EPF: published day-ahead forecasts; ISO-NE: declared
                    weather proxy at D and observed load at D-1 / D-7 only)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoLarsIC

from .foundation import COVARIATE_MODELS, FOUNDATION_MODELS, predict_price_days

PRICE_LAGS = (1, 2, 3, 7)
EXOGENOUS_LAGS = (0, 1, 7)
INFORMATION_SETS = ("price_only", "price_exogenous")
PRICE_MODELS = ("naive", "lear", "hist_gradient_boosting", "extra_trees")


# --------------------------------------------------------------------------
# Data loading: every loader returns an hourly frame with a "Price" column and
# a dict mapping exogenous column -> available lags (0 = day D itself).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PriceData:
    hourly: pd.DataFrame
    exogenous_lags: dict[str, tuple[int, ...]]
    description: dict[str, str]


def load_epftoolbox_csv(path: str | Path) -> PriceData:
    """Official epftoolbox market file: Date, Price, Exogenous 1, Exogenous 2.

    The two exogenous columns are day-ahead forecasts (load / generation) and
    are therefore legitimately known for day D at submission time.
    """
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "ts"
    # Raw files carry market-specific headers (e.g. PJM "Zonal COMED price",
    # "System load forecast"); epftoolbox renames them positionally.
    original = [str(c).strip() for c in frame.columns]
    frame.columns = ["Price"] + [f"Exogenous {i}" for i in range(1, len(original))]
    exogenous = list(frame.columns[1:])
    return PriceData(
        frame[["Price", *exogenous]].astype(float),
        {c: EXOGENOUS_LAGS for c in exogenous},
        {c: f"published day-ahead forecast ({o})" for c, o in zip(exogenous, original[1:])},
    )


def load_long_price_csv(path: str | Path, series: str) -> PriceData:
    """Long ``unique_id, ds, y`` file (price only); used for public mirrors."""
    frame = pd.read_csv(path, parse_dates=["ds"])
    frame = frame[frame["unique_id"] == series].set_index("ds")[["y"]]
    frame.index.name = "ts"
    return PriceData(frame.rename(columns={"y": "Price"}).astype(float), {}, {})


def load_isone_price(paths: list[str | Path]) -> PriceData:
    """ISO-NE system (ISO NE CA) DA LMP on a 24-hour local market-day clock.

    Day-ahead auctions are cleared per local market day, so price days are
    indexed on the US/Eastern wall clock with the usual EPF convention of 24
    values per day: the missing spring hour is linearly interpolated and the
    repeated autumn hour ("02X") is averaged with "02". The legacy 2023 file
    already has 24 rows per day and is used as published.

    Exogenous: system Dry_Bulb / Dew_Point at D (declared weather proxy) and
    RT_Demand at D-1 / D-7 (observed before submission). DA_Demand is excluded.
    """
    columns = ["DA_LMP", "RT_Demand", "Dry_Bulb", "Dew_Point"]
    frames = [
        pd.read_excel(path, sheet_name="ISO NE CA", usecols=["Date", "Hr_End", *columns])
        for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    hour_ending = pd.to_numeric(frame["Hr_End"].astype(str).str.strip().str.replace("X", ""))
    frame["ts"] = pd.to_datetime(frame["Date"]) + pd.to_timedelta(hour_ending - 1, unit="h")
    hourly = frame.groupby("ts")[columns].mean()          # averages 02 and 02X
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    filled = int(len(full.difference(hourly.index)))       # missing spring hours
    hourly = hourly.reindex(full).interpolate(limit=2).rename(columns={"DA_LMP": "Price"})
    hourly.index.name = "ts"
    return PriceData(
        hourly,
        {"Dry_Bulb": (0,), "Dew_Point": (0,), "RT_Demand": (1, 7)},
        {
            "Dry_Bulb": "observed day-D temperature (declared weather-forecast proxy)",
            "Dew_Point": "observed day-D dew point (declared weather-forecast proxy)",
            "RT_Demand": "observed load, D-1 and D-7 only",
            "dst_policy": f"24 h local market day; {filled} spring hours interpolated, 02X averaged",
        },
    )


def load_price_data(dataset: dict, root: Path) -> PriceData:
    fmt = dataset["format"]
    resolve = lambda p: Path(p) if Path(p).is_absolute() else root / p  # noqa: E731
    if fmt == "epftoolbox":
        return load_epftoolbox_csv(resolve(dataset["path"]))
    if fmt == "long_price":
        return load_long_price_csv(resolve(dataset["path"]), dataset["series"])
    if fmt == "isone_price":
        return load_isone_price([resolve(p) for p in dataset["paths"]])
    raise ValueError(f"Unknown price dataset format: {fmt}")


# --------------------------------------------------------------------------
# Day x hour design matrix
# --------------------------------------------------------------------------
def daily_matrix(series: pd.Series) -> pd.DataFrame:
    """Rows = calendar day, columns = hour 0..23 (requires a regular hourly index)."""
    frame = series.to_frame("v")
    frame["day"] = frame.index.normalize()
    frame["hour"] = frame.index.hour
    matrix = frame.pivot_table(index="day", columns="hour", values="v", aggfunc="mean")
    complete = matrix.notna().sum(axis=1) == 24
    return matrix.loc[complete]


def build_design(data: PriceData, information_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, Y) indexed by delivery day D. Every column is known on D-1."""
    if information_set not in INFORMATION_SETS:
        raise ValueError(f"Unknown information set: {information_set}")
    price = daily_matrix(data.hourly["Price"])
    days = pd.date_range(price.index.min(), price.index.max(), freq="D")
    price = price.reindex(days)
    blocks = []
    for lag in PRICE_LAGS:
        shifted = price.shift(lag)
        shifted.columns = [f"price_d{lag}_h{h}" for h in range(24)]
        blocks.append(shifted)
    if information_set == "price_exogenous":
        for column, lags in data.exogenous_lags.items():
            matrix = daily_matrix(data.hourly[column]).reindex(days)
            for lag in lags:
                shifted = matrix.shift(lag)
                shifted.columns = [f"{column}_d{lag}_h{h}" for h in range(24)]
                blocks.append(shifted)
    dow = pd.get_dummies(pd.Series(days.dayofweek, index=days), prefix="dow").astype(float)
    x = pd.concat([*blocks, dow], axis=1)
    y = price.copy()
    y.columns = [f"h{h}" for h in range(24)]
    usable = x.notna().all(axis=1) & y.notna().all(axis=1)
    return x.loc[usable], y.loc[usable]


# --------------------------------------------------------------------------
# Variance-stabilising transform (Uniejewski, Weron & Ziel 2018): median/MAD
# standardisation over the calibration window followed by asinh. Applied to
# every model family so that the comparison isolates the learner.
# --------------------------------------------------------------------------
class AsinhScaler:
    def fit(self, values: np.ndarray) -> "AsinhScaler":
        self.median = np.median(values, axis=0)
        mad = stats.median_abs_deviation(values, axis=0, scale="normal")
        self.mad = np.where(mad > 1e-9, mad, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.arcsinh((values - self.median) / self.mad)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.sinh(values) * self.mad + self.median


def naive_forecast(y: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
    """epftoolbox naive: D-7 for Monday, Saturday, Sunday; D-1 otherwise."""
    lag = 7 if day.dayofweek in (0, 5, 6) else 1
    source = day - pd.Timedelta(days=lag)
    return y.loc[source].to_numpy() if source in y.index else np.full(24, np.nan)


def _fit_predict(model: str, x_train, y_train, x_test, params: dict, seed: int) -> np.ndarray:
    """One model per delivery hour (24 models), all in transformed space."""
    x_scaler = AsinhScaler().fit(x_train)
    y_scaler = AsinhScaler().fit(y_train)
    xt, xs = x_scaler.transform(x_train), x_scaler.transform(x_test)
    yt = y_scaler.transform(y_train)
    dummies = np.array([c.startswith("dow_") for c in params["_columns"]])
    # day-of-week dummies are left untransformed
    xt[:, dummies], xs[:, dummies] = x_train[:, dummies], x_test[:, dummies]
    out = np.zeros((len(x_test), 24))
    for hour in range(24):
        if model == "lear":
            # LassoLarsIC picks alpha by AIC; the final fit uses coordinate-descent
            # Lasso with that alpha, as in the epftoolbox reference implementation.
            alpha = LassoLarsIC(criterion="aic", max_iter=2500).fit(xt, yt[:, hour]).alpha_
            estimator = Lasso(alpha=alpha, max_iter=5000)
        elif model == "hist_gradient_boosting":
            estimator = HistGradientBoostingRegressor(**params.get("hgb", {}), random_state=seed)
        elif model == "extra_trees":
            estimator = ExtraTreesRegressor(**params.get("et", {}), n_jobs=-1, random_state=seed)
        else:
            raise ValueError(model)
        out[:, hour] = estimator.fit(xt, yt[:, hour]).predict(xs)
    return y_scaler.inverse(out)


def rolling_forecast(
    x: pd.DataFrame,
    y: pd.DataFrame,
    test_days: pd.DatetimeIndex,
    model: str,
    calibration_days: int | None,
    recalibration_days: int,
    params: dict,
    seed: int,
) -> pd.DataFrame:
    """Recalibrate every ``recalibration_days`` and forecast the following block.

    The calibration window for block starting at day D is
    [D - calibration_days, D - 1] (or everything before D when None).
    """
    predictions = []
    params = {**params, "_columns": list(x.columns)}
    for start in range(0, len(test_days), recalibration_days):
        block = test_days[start : start + recalibration_days]
        first = block[0]
        if model == "naive":
            predictions.append(
                pd.DataFrame([naive_forecast(y, d) for d in block], index=block, columns=y.columns)
            )
            continue
        train_mask = x.index < first
        if calibration_days is not None:
            train_mask &= x.index >= first - pd.Timedelta(days=calibration_days)
        block_in_x = block.intersection(x.index)
        values = _fit_predict(
            model,
            x.loc[train_mask].to_numpy(dtype=float),
            y.loc[train_mask].to_numpy(dtype=float),
            x.loc[block_in_x].to_numpy(dtype=float),
            params,
            seed,
        )
        predictions.append(pd.DataFrame(values, index=block_in_x, columns=y.columns))
        print(f"  {model}: recalibrated on {int(train_mask.sum())} days, forecast "
              f"{block[0].date()}..{block[-1].date()}", flush=True)
    return pd.concat(predictions)


# --------------------------------------------------------------------------
# Metrics and the multivariate Diebold-Mariano test (Ziel & Weron 2018), as
# implemented in epftoolbox: per-day loss = ||error vector||_1, one-sided.
# --------------------------------------------------------------------------
def price_metrics(actual: pd.DataFrame, predicted: pd.DataFrame, naive: pd.DataFrame) -> dict:
    a, p, n = actual.to_numpy(), predicted.to_numpy(), naive.to_numpy()
    e = a - p
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "smape_pct": float(100 * np.mean(2 * np.abs(e) / (np.abs(a) + np.abs(p) + 1e-9))),
        "rmae": float(np.abs(e).mean() / np.abs(a - n).mean()),
    }


def multivariate_dm(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, norm: int = 1):
    """H0: forecast A is not less accurate than B. Small p => B significantly better than A."""
    loss_a = np.linalg.norm(actual - pred_a, ord=norm, axis=1)
    loss_b = np.linalg.norm(actual - pred_b, ord=norm, axis=1)
    d = loss_a - loss_b
    statistic = d.mean() / np.sqrt(d.var(ddof=0) / len(d))
    return float(statistic), float(1 - stats.norm.cdf(statistic))


def run_price_experiment(config: dict, models: list[str], information_sets: list[str], output: Path,
                         root: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    data = load_price_data(config["dataset"], root)
    test_cfg = config["test"]
    seed = config.get("seed", 42)
    rows, preds, timings = [], {}, {}
    for info in information_sets:
        if info == "price_exogenous" and not data.exogenous_lags:
            print(f"[skip] {info}: dataset has no exogenous inputs")
            continue
        x, y = build_design(data, info)
        test_days = y.index[(y.index >= test_cfg["start"]) & (y.index <= test_cfg["end"])]
        naive = rolling_forecast(x, y, test_days, "naive", None, 1, {}, seed)
        for model in models:
            if model == "naive" and info != information_sets[0]:
                continue
            began = time.time()
            cache = output / "forecasts" / f"{model}__{info}.csv"
            if model != "naive" and cache.exists():
                forecast = pd.read_csv(cache, index_col=0, parse_dates=True)
                forecast.columns = y.columns
                print(f"  {model}|{info}: loaded cached forecasts", flush=True)
            elif model == "naive":
                forecast = naive
            elif model in FOUNDATION_MODELS:
                if info == "price_exogenous" and model not in COVARIATE_MODELS:
                    continue
                values = predict_price_days(model, data.hourly, test_days, data.exogenous_lags,
                                            info == "price_exogenous", config.get("foundation", {}))
                forecast = pd.DataFrame(values, index=test_days, columns=y.columns)
            else:
                every = config.get("recalibration_days", 7)
                if isinstance(every, dict):  # per-model recalibration frequency
                    every = every.get(model, every.get("default", 7))
                forecast = rolling_forecast(
                    x, y, test_days, model, config.get("calibration_days"),
                    every, config.get("models", {}), seed,
                )
            if model != "naive" and not cache.exists():
                cache.parent.mkdir(exist_ok=True)
                forecast.to_csv(cache)
            timings[(model, info)] = time.time() - began
            common = forecast.index.intersection(naive.index)
            preds[(model, info)] = forecast.loc[common]
            rows.append({"model": model, "information_set": info if model != "naive" else "-",
                         "test_days": len(common),
                         **price_metrics(y.loc[common], forecast.loc[common], naive.loc[common]),
                         "seconds": round(timings[(model, info)], 1)})
            print(rows[-1], flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "price_metrics.csv", index=False)

    # long-form predictions for every plotted / tested point
    long = []
    for (model, info), frame in preds.items():
        stacked = frame.stack().rename("predicted").reset_index()
        stacked.columns = ["day", "hour", "predicted"]
        stacked["model"], stacked["information_set"] = model, info
        long.append(stacked)
    actual = y.stack().rename("actual").reset_index()
    actual.columns = ["day", "hour", "actual"]
    long = pd.concat(long).merge(actual, on=["day", "hour"], how="left")
    long.to_csv(output / "price_predictions.csv", index=False)

    # pairwise multivariate DM p-values on the common day set
    keys = list(preds)
    common = preds[keys[0]].index
    for key in keys[1:]:
        common = common.intersection(preds[key].index)
    truth = y.loc[common].to_numpy()
    names = [f"{m}|{i}" if m != "naive" else "naive" for m, i in keys]
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    for ka, na in zip(keys, names):
        for kb, nb in zip(keys, names):
            if ka != kb:
                table.loc[na, nb] = multivariate_dm(truth, preds[ka].loc[common].to_numpy(),
                                                    preds[kb].loc[common].to_numpy())[1]
    table.to_csv(output / "dm_pvalues.csv")
    (output / "price_manifest.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in config.items()]
                  + [f"exogenous: {data.description}", f"common test days: {len(common)}"])
    )
    return metrics
