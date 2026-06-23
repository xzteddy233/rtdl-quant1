from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeutralizationSummary:
    rows: int
    valid_rows: int
    industry_coverage: float
    market_cap_coverage: float


def load_industry_map(
    path: str | Path,
    *,
    code_column: str = "symbol",
    industry_column: str = "industry",
) -> pd.DataFrame:
    """Load one static industry classification per stock code."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = {code_column, industry_column}.difference(frame.columns)
    if missing:
        raise KeyError(f"Industry file is missing columns: {sorted(missing)}")
    result = frame[[code_column, industry_column]].copy()
    result.columns = ["code", "industry"]
    result["code"] = result["code"].astype(str).str.upper()
    result["industry"] = result["industry"].astype("string")
    return result.drop_duplicates("code", keep="last")


def load_float_market_cap(
    prices_dir: str | Path,
    codes: Iterable[str],
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Estimate daily free-float market cap from close, volume and turnover.

    ``turn`` is stored in percent, so free-float shares are approximately
    ``volume / (turn / 100)`` and market cap is close times that share count.
    """
    root = Path(prices_dir)
    start = None if start_date is None else pd.Timestamp(start_date)
    end = None if end_date is None else pd.Timestamp(end_date)
    frames: list[pd.DataFrame] = []
    for code in sorted(set(str(code).upper() for code in codes)):
        path = root / f"{code}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=["date", "symbol", "close", "volume", "turn"],
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("close", "volume", "turn"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if start is not None:
            frame = frame.loc[frame["date"].ge(start)]
        if end is not None:
            frame = frame.loc[frame["date"].le(end)]
        turnover_fraction = frame["turn"] / 100.0
        frame["float_market_cap"] = (
            frame["close"] * frame["volume"] / turnover_fraction
        )
        frame.loc[
            turnover_fraction.le(0)
            | frame["close"].le(0)
            | frame["volume"].le(0),
            "float_market_cap",
        ] = np.nan
        frame["code"] = frame["symbol"].astype(str).str.upper()
        frames.append(frame[["date", "code", "float_market_cap"]])
    if not frames:
        return pd.DataFrame(columns=["date", "code", "float_market_cap"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["date", "code"], keep="last"
    )


def neutralize_cross_sectional_signal(
    frame: pd.DataFrame,
    *,
    signal_column: str = "raw_prediction",
    output_column: str = "prediction",
    date_column: str = "date",
    industry_column: str = "industry",
    market_cap_column: str = "float_market_cap",
    neutralize_industry: bool = True,
    neutralize_market_cap: bool = True,
    standardize: bool = True,
) -> tuple[pd.DataFrame, NeutralizationSummary]:
    """Residualize a daily signal against industry and log free-float cap.

    Industry effects are removed by within-industry demeaning. The market-cap
    slope is then estimated on industry-demeaned log cap (Frisch-Waugh-Lovell),
    which is algebraically equivalent to OLS on industry dummies and log cap.
    """
    required = {date_column, signal_column}
    if neutralize_industry:
        required.add(industry_column)
    if neutralize_market_cap:
        required.add(market_cap_column)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Neutralization frame is missing: {sorted(missing)}")

    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    signal = pd.to_numeric(result[signal_column], errors="coerce")
    industry = result.get(
        industry_column, pd.Series("UNKNOWN", index=result.index)
    ).fillna("UNKNOWN")
    market_cap = pd.to_numeric(
        result.get(market_cap_column, np.nan), errors="coerce"
    )
    log_market_cap = np.log(market_cap.where(market_cap.gt(0)))

    neutralized = pd.Series(np.nan, index=result.index, dtype=float)
    for _, indices in result.groupby(date_column, sort=False).groups.items():
        index = pd.Index(indices)
        y = signal.loc[index]
        valid = y.notna()
        if neutralize_market_cap:
            valid &= log_market_cap.loc[index].notna()
        if not valid.any():
            continue
        valid_index = index[valid.to_numpy()]
        residual = y.loc[valid_index].astype(float)

        if neutralize_industry:
            groups = industry.loc[valid_index]
            residual = residual - residual.groupby(groups).transform("mean")

        if neutralize_market_cap:
            x = log_market_cap.loc[valid_index].astype(float)
            if neutralize_industry:
                x = x - x.groupby(industry.loc[valid_index]).transform("mean")
            else:
                x = x - x.mean()
                residual = residual - residual.mean()
            denominator = float(np.square(x).sum())
            if denominator > 1e-12:
                beta = float((x * residual).sum() / denominator)
                residual = residual - beta * x

        if standardize:
            std = float(residual.std(ddof=0))
            residual = (
                residual - residual.mean()
                if std <= 1e-12
                else (residual - residual.mean()) / std
            )
        neutralized.loc[valid_index] = residual

    result[output_column] = neutralized
    summary = NeutralizationSummary(
        rows=len(result),
        valid_rows=int(neutralized.notna().sum()),
        industry_coverage=float(result[industry_column].notna().mean())
        if industry_column in result
        else 0.0,
        market_cap_coverage=float(market_cap.gt(0).mean()),
    )
    return result, summary
