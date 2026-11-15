"""Forecast-comparison tests used in the paper.

* ``dm_test``           Diebold-Mariano on squared-error loss differentials with
                        a Newey-West (Bartlett) long-run variance and the
                        Harvey-Leybourne-Newbold small-sample correction. The
                        default bandwidth is one week of hourly origins (168),
                        because hourly loss differentials of 24-h-ahead
                        forecasts are autocorrelated at daily and weekly lags,
                        not only up to h-1.
* ``daily_dm_test``     the same test on loss differentials averaged per
                        target day (weekly bandwidth on the daily series): a
                        robustness check that removes intraday dependence.
* ``multivariate_dm``   the multivariate DM test of Ziel & Weron (2018) on the
                        L1 norm of the 24-h error vector, one-sided; with
                        ``lags > 0`` the variance is Newey-West instead of the
                        i.i.d. variance used by epftoolbox.
* ``holm``              Holm-Bonferroni adjustment for a family of p-values.
* ``window_test``       Wilcoxon signed-rank test of per-window errors of two
                        methods across rolling-origin windows.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _newey_west_variance(d: np.ndarray, lags: int) -> float:
    n = len(d)
    dc = d - d.mean()
    lags = int(min(max(lags, 0), n - 1))
    gamma0 = float(np.dot(dc, dc) / n)
    total = gamma0
    for k in range(1, lags + 1):
        total += 2 * (1 - k / (lags + 1)) * float(np.dot(dc[k:], dc[: n - k]) / n)
    return max(total, 1e-12)


def dm_test(e_model: np.ndarray, e_bench: np.ndarray, horizon: int = 24,
            lags: int = 168) -> tuple[float, float]:
    """Two-sided DM test. Negative statistic: the model has smaller squared error."""
    d = np.asarray(e_model, float) ** 2 - np.asarray(e_bench, float) ** 2
    n = len(d)
    stat = d.mean() / np.sqrt(_newey_west_variance(d, lags) / n)
    h = horizon
    stat *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def daily_dm_test(e_model: np.ndarray, e_bench: np.ndarray, target_days: np.ndarray,
                  lags: int = 7) -> tuple[float, float]:
    """DM test on the mean squared-error differential of each target day."""
    d = np.asarray(e_model, float) ** 2 - np.asarray(e_bench, float) ** 2
    days = np.asarray(target_days)
    order = np.unique(days)
    daily = np.array([d[days == day].mean() for day in order])
    n = len(daily)
    stat = daily.mean() / np.sqrt(_newey_west_variance(daily, lags) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def multivariate_dm(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    norm: int = 1, lags: int = 0) -> tuple[float, float]:
    """H0: forecast A is not less accurate than B. Small p => B significantly better."""
    loss_a = np.linalg.norm(actual - pred_a, ord=norm, axis=1)
    loss_b = np.linalg.norm(actual - pred_b, ord=norm, axis=1)
    d = loss_a - loss_b
    variance = _newey_west_variance(d, lags) if lags > 0 else d.var(ddof=0)
    statistic = d.mean() / np.sqrt(variance / len(d))
    return float(statistic), float(1 - stats.norm.cdf(statistic))


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def window_test(errors_a, errors_b) -> tuple[int, float]:
    """Wilcoxon signed-rank test across windows. Returns (#windows A better, two-sided p)."""
    a, b = np.asarray(errors_a, float), np.asarray(errors_b, float)
    wins = int((a < b).sum())
    if np.allclose(a, b):
        return wins, 1.0
    return wins, float(stats.wilcoxon(a, b).pvalue)
