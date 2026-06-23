import numpy as np

from rtdl_quant.metrics import ic, icir, mae, mse, rank_ic, rmse


def test_regression_metrics() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 3.0, 2.0])
    assert mse(actual, predicted) == 2 / 3
    assert np.isclose(rmse(actual, predicted), np.sqrt(2 / 3))
    assert mae(actual, predicted) == 2 / 3


def test_information_coefficients() -> None:
    actual = np.array([1.0, 2.0, 3.0, np.nan])
    predicted = np.array([2.0, 4.0, 6.0, 10.0])
    assert np.isclose(ic(actual, predicted), 1.0)
    assert np.isclose(rank_ic(actual, predicted), 1.0)


def test_icir_uses_sample_standard_deviation() -> None:
    values = np.array([0.1, 0.2, 0.3])
    assert np.isclose(icir(values), values.mean() / values.std(ddof=1))


def test_constant_signal_has_undefined_ic() -> None:
    assert np.isnan(ic([1, 2, 3], [1, 1, 1]))
