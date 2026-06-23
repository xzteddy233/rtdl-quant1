from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from rtdl_quant.metrics import ic, icir, rank_ic


@dataclass(frozen=True)
class ICSummary:
    ic_mean: float
    rank_ic_mean: float
    icir: float
    rank_icir: float
    observations: int


class ICAnalysis:
    """Daily cross-sectional Pearson IC and Spearman RankIC analysis."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        date_column: str = "date",
        code_column: str = "code",
        prediction_column: str = "prediction",
        label_column: str = "label",
    ) -> None:
        required = {date_column, code_column, prediction_column, label_column}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")
        self.date_column = date_column
        self.prediction_column = prediction_column
        self.label_column = label_column
        self.frame = frame.copy()
        self.frame[date_column] = pd.to_datetime(self.frame[date_column])

    def daily(self) -> pd.DataFrame:
        def calculate(group: pd.DataFrame) -> pd.Series:
            label = group[self.label_column].to_numpy()
            prediction = group[self.prediction_column].to_numpy()
            return pd.Series(
                {"ic": ic(label, prediction), "rank_ic": rank_ic(label, prediction)}
            )

        result = self.frame.groupby(self.date_column, sort=True)[
            [self.label_column, self.prediction_column]
        ].apply(calculate)
        result.index.name = self.date_column
        result["cumulative_ic"] = result["ic"].fillna(0.0).cumsum()
        result["cumulative_rank_ic"] = result["rank_ic"].fillna(0.0).cumsum()
        return result

    def summary(self, daily: pd.DataFrame | None = None) -> ICSummary:
        daily = self.daily() if daily is None else daily
        return ICSummary(
            ic_mean=float(daily["ic"].mean()),
            rank_ic_mean=float(daily["rank_ic"].mean()),
            icir=icir(daily["ic"]),
            rank_icir=icir(daily["rank_ic"]),
            observations=len(daily),
        )

    def result(self) -> pd.DataFrame:
        """Return daily values with aggregate statistics repeated as columns."""
        daily = self.daily()
        summary = self.summary(daily)
        daily["ic_mean"] = summary.ic_mean
        daily["rank_ic_mean"] = summary.rank_ic_mean
        daily["icir"] = summary.icir
        daily["rank_icir"] = summary.rank_icir
        return daily

    def plot(
        self,
        daily: pd.DataFrame | None = None,
        *,
        ax: Axes | None = None,
        include_rank_ic: bool = True,
    ) -> Axes:
        daily = self.daily() if daily is None else daily
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 5))
        ax.plot(daily.index, daily["cumulative_ic"], label="Cumulative IC")
        if include_rank_ic:
            ax.plot(
                daily.index,
                daily["cumulative_rank_ic"],
                label="Cumulative RankIC",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(title="Cumulative Information Coefficient", xlabel="Date", ylabel="IC sum")
        ax.grid(alpha=0.25)
        ax.legend()
        return ax
