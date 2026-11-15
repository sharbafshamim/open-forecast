import numpy as np

from open_forecast.significance import daily_dm_test, dm_test, holm, multivariate_dm, window_test


def test_dm_detects_better_model_and_is_symmetric():
    rng = np.random.default_rng(0)
    bench = rng.normal(0, 2.0, 2000)
    model = rng.normal(0, 1.0, 2000)
    stat, p = dm_test(model, bench)
    assert stat < 0 and p < 1e-3
    stat2, p2 = dm_test(bench, model)
    assert np.isclose(stat, -stat2) and np.isclose(p, p2)


def test_longer_bandwidth_is_more_conservative_under_persistent_losses():
    rng = np.random.default_rng(1)
    n = 24 * 120
    d = np.empty(n)
    d[0] = 0.0
    for i in range(1, n):                      # strongly persistent loss differential
        d[i] = 0.995 * d[i - 1] + rng.normal(0, 0.1)
    d += 0.3
    e_bench = np.full(n, 3.0)
    e_model = np.sqrt(e_bench ** 2 + d)
    _, p_short = dm_test(e_model, e_bench, lags=23)
    _, p_long = dm_test(e_model, e_bench, lags=168)
    assert p_long > p_short
    _, p_daily = daily_dm_test(e_model, e_bench, np.arange(n) // 24)
    assert 0 <= p_daily <= 1


def test_holm_adjustment():
    adjusted = holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adjusted, [0.04, 0.09, 0.09, 0.5])


def test_multivariate_dm_newey_west_and_window_test():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(300, 24))
    good, bad = truth + rng.normal(0, 0.5, truth.shape), truth + rng.normal(0, 1.0, truth.shape)
    assert multivariate_dm(truth, bad, good)[1] < 0.01
    assert multivariate_dm(truth, bad, good, lags=7)[1] < 0.01
    wins, p = window_test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert wins == 11 and p < 0.01
