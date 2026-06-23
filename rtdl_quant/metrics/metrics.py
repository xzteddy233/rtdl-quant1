from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata


def _paired_finite(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} != {pred.shape}")
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        raise ValueError("No finite paired observations")
    return true[mask], pred[mask]


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error over finite paired observations."""
    true, pred = _paired_finite(y_true, y_pred)
    return float(np.mean(np.square(pred - true)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error over finite paired observations."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error over finite paired observations."""
    true, pred = _paired_finite(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def ic(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Pearson information coefficient."""
    true, pred = _paired_finite(y_true, y_pred)
    if len(true) < 2 or np.isclose(true.std(), 0.0) or np.isclose(pred.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def rank_ic(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Spearman information coefficient with average ranks for ties."""
    true, pred = _paired_finite(y_true, y_pred)
    return ic(rankdata(true, method="average"), rankdata(pred, method="average"))


def icir(ic_values: Iterable[float], annualization: float | None = None) -> float:
    """Information coefficient information ratio.

    By default this returns ``mean(IC) / sample_std(IC)``. Pass an
    annualization factor such as ``252`` to multiply the ratio by its square
    root.
    """
    values = np.asarray(list(ic_values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    std = values.std(ddof=1)
    if np.isclose(std, 0.0):
        return float("nan")
    ratio = values.mean() / std
    if annualization is not None:
        if annualization <= 0:
            raise ValueError("annualization must be positive")
        ratio *= np.sqrt(annualization)
    return float(ratio)


if __name__ == "__main__":
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([0.8, 2.2, 2.7, 4.1])
    print(
        {
            "mse": mse(actual, predicted),
            "rmse": rmse(actual, predicted),
            "mae": mae(actual, predicted),
            "ic": ic(actual, predicted),
            "rank_ic": rank_ic(actual, predicted),
        }
    )
