from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


@dataclass(frozen=True)
class GroupBacktestResult:
    group_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    long_short_returns: pd.Series
    cumulative_long_short: pd.Series
    top_bottom_spread: pd.Series


class GroupBacktest:
    """Equal-weight daily quantile portfolio backtest."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        n_groups: int = 10,
        date_column: str = "date",
        prediction_column: str = "prediction",
        return_column: str = "future_return",
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        required = {date_column, prediction_column, return_column}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")
        self.n_groups = n_groups
        self.date_column = date_column
        self.prediction_column = prediction_column
        self.return_column = return_column
        self.frame = frame.copy()
        self.frame[date_column] = pd.to_datetime(self.frame[date_column])

    def run(self) -> GroupBacktestResult:
        data = self.frame.dropna(
            subset=[self.date_column, self.prediction_column, self.return_column]
        ).copy()

        by_date = data.groupby(self.date_column, sort=False)
        # First-rank ties deterministically, then map each rank to an equal-count
        # bucket. This is equivalent to qcut without duplicate-edge failures.
        ranks = by_date[self.prediction_column].rank(method="first")
        counts = by_date[self.prediction_column].transform("size")
        groups = np.ceil(ranks * self.n_groups / counts).clip(1, self.n_groups)
        data["group"] = groups.where(counts >= self.n_groups)
        grouped = data
        grouped = grouped.dropna(subset=["group"])
        group_returns = (
            grouped.groupby([self.date_column, "group"])[self.return_column]
            .mean()
            .unstack("group")
            .sort_index()
            .reindex(columns=range(1, self.n_groups + 1))
        )
        group_returns.columns = [f"group_{i}" for i in range(1, self.n_groups + 1)]
        cumulative = (1.0 + group_returns.fillna(0.0)).cumprod() - 1.0

        bottom = group_returns["group_1"]
        top = group_returns[f"group_{self.n_groups}"]
        spread = top - bottom
        long_short = spread.rename("long_short_return")
        cumulative_long_short = ((1.0 + long_short.fillna(0.0)).cumprod() - 1.0).rename(
            "cumulative_long_short"
        )
        return GroupBacktestResult(
            group_returns=group_returns,
            cumulative_returns=cumulative,
            long_short_returns=long_short,
            cumulative_long_short=cumulative_long_short,
            top_bottom_spread=spread.rename("top_bottom_spread"),
        )

    @staticmethod
    def plot_cumulative(
        result: GroupBacktestResult, *, ax: Axes | None = None
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 6))
        result.cumulative_returns.plot(ax=ax, alpha=0.75)
        ax.set(
            title="Cumulative Return by Prediction Group",
            xlabel="Date",
            ylabel="Cumulative return",
        )
        ax.grid(alpha=0.25)
        return ax

    @staticmethod
    def plot_long_short(
        result: GroupBacktestResult, *, ax: Axes | None = None
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 5))
        result.cumulative_long_short.plot(ax=ax, label="Top - Bottom")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(
            title="Cumulative Long-Short Return",
            xlabel="Date",
            ylabel="Cumulative return",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        return ax
