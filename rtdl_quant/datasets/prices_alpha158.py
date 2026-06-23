from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-12
ROLLING_WINDOWS = (5, 10, 20, 30, 60)
ROLLING_OPERATORS = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)
KBAR_FEATURES = (
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
)
PRICE_FEATURES = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
ALPHA158_FEATURES = (
    *KBAR_FEATURES,
    *PRICE_FEATURES,
    *(f"{operator}{window}" for operator in ROLLING_OPERATORS for window in ROLLING_WINDOWS),
)

if len(ALPHA158_FEATURES) != 158:
    raise RuntimeError("Alpha158 feature definition must contain exactly 158 columns")


@dataclass(frozen=True)
class PricesBuildConfig:
    prices_dir: str | Path = "prices"
    output_path: str | Path = "data/alpha158_prices.parquet"
    start_date: str | None = None
    end_date: str | None = None
    horizon: int = 20
    exclude_st: bool = True
    require_trading: bool = True
    max_instruments: int | None = None


class PricesAlpha158Builder:
    """Build Qlib-compatible Alpha158 factors from per-stock OHLCV CSV files.

    Files are processed one instrument at a time, so raw input size does not
    determine peak memory. The final panel is cached as Parquet for experiments.
    """

    required_columns = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "tradestatus",
        "isST",
    }

    def __init__(self, config: PricesBuildConfig) -> None:
        self.config = config
        if config.horizon <= 0:
            raise ValueError("horizon must be positive")

    def files(self) -> list[Path]:
        paths = sorted(Path(self.config.prices_dir).glob("*.csv"))
        if not paths:
            raise FileNotFoundError(
                f"No CSV files found under {Path(self.config.prices_dir).resolve()}"
            )
        if self.config.max_instruments is not None:
            if self.config.max_instruments <= 0:
                raise ValueError("max_instruments must be positive")
            count = min(self.config.max_instruments, len(paths))
            indices = np.linspace(0, len(paths) - 1, count, dtype=int)
            paths = [paths[index] for index in indices]
        return paths

    def build_frame(self, files: Iterable[Path] | None = None) -> pd.DataFrame:
        """Build an in-memory panel; intended for tests and small universes."""
        frames = [self.transform_file(path) for path in (files or self.files())]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            raise ValueError("No valid observations were produced")
        panel = pd.concat(frames, ignore_index=True)
        return self._add_rank_label(panel)

    def build_to_parquet(self) -> Path:
        """Build the configured universe and save a reusable Parquet cache."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        output = Path(self.config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        writer: pq.ParquetWriter | None = None
        rows_written = 0
        try:
            paths = self.files()
            for index, path in enumerate(paths, start=1):
                frame = self.transform_file(path)
                if frame.empty:
                    continue
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="zstd",
                        use_dictionary=["code"],
                    )
                writer.write_table(table)
                rows_written += len(frame)
                if index == 1 or index % 100 == 0 or index == len(paths):
                    LOGGER.info(
                        "processed_instruments=%d/%d rows_written=%d",
                        index,
                        len(paths),
                        rows_written,
                    )
        except Exception:
            if writer is not None:
                writer.close()
                writer = None
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.close()
        if rows_written == 0:
            temporary.unlink(missing_ok=True)
            raise ValueError("No valid observations were produced")
        temporary.replace(output)
        return output.resolve()

    def transform_file(self, path: str | Path) -> pd.DataFrame:
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        missing = self.required_columns.difference(raw.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")

        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "tradestatus",
            "isST",
        ]
        raw[numeric_columns] = raw[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        raw = raw.dropna(subset=["date", "open", "high", "low", "close"])
        raw = raw.sort_values("date").drop_duplicates("date", keep="last")
        if raw.empty:
            return pd.DataFrame()

        factors = compute_alpha158(raw)
        future_close = raw["close"].shift(-self.config.horizon)
        factors["future_return"] = future_close / raw["close"] - 1.0
        factors["date"] = raw["date"].to_numpy()
        factors["code"] = raw["symbol"].astype(str).to_numpy()

        mask = pd.Series(True, index=raw.index)
        if self.config.require_trading:
            mask &= raw["tradestatus"].eq(1)
        if self.config.exclude_st:
            mask &= raw["isST"].eq(0)
        if self.config.start_date is not None:
            mask &= raw["date"].ge(pd.Timestamp(self.config.start_date))
        if self.config.end_date is not None:
            mask &= raw["date"].le(pd.Timestamp(self.config.end_date))

        columns = ["date", "code", *ALPHA158_FEATURES, "future_return"]
        result = factors.loc[mask, columns].replace([np.inf, -np.inf], np.nan)
        result = result.dropna(subset=[*ALPHA158_FEATURES, "future_return"])
        result.loc[:, ALPHA158_FEATURES] = result.loc[
            :, ALPHA158_FEATURES
        ].astype(np.float32)
        result["future_return"] = result["future_return"].astype(np.float32)
        return result.reset_index(drop=True)

    @staticmethod
    def _add_rank_label(panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()
        panel["label"] = panel.groupby("date")["future_return"].rank(
            method="average", pct=True
        )
        return panel.sort_values(["date", "code"]).reset_index(drop=True)


def compute_alpha158(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the 158 default factors defined by Qlib's ``Alpha158DL``."""
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    vwap = (high + low + close) / 3.0
    spread = high - low

    output: dict[str, pd.Series] = {}
    output["KMID"] = (close - open_) / (open_ + EPSILON)
    output["KLEN"] = spread / (open_ + EPSILON)
    output["KMID2"] = (close - open_) / (spread + EPSILON)
    output["KUP"] = (high - np.maximum(open_, close)) / (open_ + EPSILON)
    output["KUP2"] = (high - np.maximum(open_, close)) / (spread + EPSILON)
    output["KLOW"] = (np.minimum(open_, close) - low) / (open_ + EPSILON)
    output["KLOW2"] = (np.minimum(open_, close) - low) / (spread + EPSILON)
    output["KSFT"] = (2.0 * close - high - low) / (open_ + EPSILON)
    output["KSFT2"] = (2.0 * close - high - low) / (spread + EPSILON)
    output["OPEN0"] = open_ / (close + EPSILON)
    output["HIGH0"] = high / (close + EPSILON)
    output["LOW0"] = low / (close + EPSILON)
    output["VWAP0"] = vwap / (close + EPSILON)

    close_change = close - close.shift(1)
    volume_change = volume - volume.shift(1)
    close_ratio = close / (close.shift(1) + EPSILON)
    volume_ratio_log = np.log(volume / (volume.shift(1) + EPSILON) + 1.0)
    weighted_volatility = np.abs(close_ratio - 1.0) * volume

    for window in ROLLING_WINDOWS:
        rolling_close = close.rolling(window, min_periods=window)
        rolling_volume = volume.rolling(window, min_periods=window)
        rolling_abs_change = close_change.abs().rolling(window, min_periods=window)
        rolling_abs_volume_change = volume_change.abs().rolling(
            window, min_periods=window
        )
        slope, rsquare, residual = _rolling_linear_stats(close, window)

        output[f"ROC{window}"] = close.shift(window) / (close + EPSILON)
        output[f"MA{window}"] = rolling_close.mean() / (close + EPSILON)
        output[f"STD{window}"] = rolling_close.std() / (close + EPSILON)
        output[f"BETA{window}"] = slope / (close + EPSILON)
        output[f"RSQR{window}"] = rsquare
        output[f"RESI{window}"] = residual / (close + EPSILON)
        output[f"MAX{window}"] = high.rolling(window).max() / (close + EPSILON)
        output[f"MIN{window}"] = low.rolling(window).min() / (close + EPSILON)
        output[f"QTLU{window}"] = rolling_close.quantile(0.8) / (close + EPSILON)
        output[f"QTLD{window}"] = rolling_close.quantile(0.2) / (close + EPSILON)
        output[f"RANK{window}"] = rolling_close.apply(_last_rank, raw=True)
        rolling_high = high.rolling(window, min_periods=window)
        rolling_low = low.rolling(window, min_periods=window)
        output[f"RSV{window}"] = (close - rolling_low.min()) / (
            rolling_high.max() - rolling_low.min() + EPSILON
        )
        index_max = rolling_high.apply(lambda values: np.argmax(values) + 1, raw=True)
        index_min = rolling_low.apply(lambda values: np.argmin(values) + 1, raw=True)
        output[f"IMAX{window}"] = index_max / window
        output[f"IMIN{window}"] = index_min / window
        output[f"IMXD{window}"] = (index_max - index_min) / window
        output[f"CORR{window}"] = close.rolling(window).corr(np.log(volume + 1.0))
        output[f"CORD{window}"] = close_ratio.rolling(window).corr(volume_ratio_log)

        up = close_change.gt(0).astype(float)
        down = close_change.lt(0).astype(float)
        output[f"CNTP{window}"] = up.rolling(window).mean()
        output[f"CNTN{window}"] = down.rolling(window).mean()
        output[f"CNTD{window}"] = output[f"CNTP{window}"] - output[f"CNTN{window}"]
        gain = close_change.clip(lower=0).rolling(window).sum()
        loss = (-close_change).clip(lower=0).rolling(window).sum()
        output[f"SUMP{window}"] = gain / (rolling_abs_change.sum() + EPSILON)
        output[f"SUMN{window}"] = loss / (rolling_abs_change.sum() + EPSILON)
        output[f"SUMD{window}"] = (gain - loss) / (
            rolling_abs_change.sum() + EPSILON
        )
        output[f"VMA{window}"] = rolling_volume.mean() / (volume + EPSILON)
        output[f"VSTD{window}"] = rolling_volume.std() / (volume + EPSILON)
        weighted = weighted_volatility.rolling(window, min_periods=window)
        output[f"WVMA{window}"] = weighted.std() / (weighted.mean() + EPSILON)
        volume_gain = volume_change.clip(lower=0).rolling(window).sum()
        volume_loss = (-volume_change).clip(lower=0).rolling(window).sum()
        absolute_volume = rolling_abs_volume_change.sum() + EPSILON
        output[f"VSUMP{window}"] = volume_gain / absolute_volume
        output[f"VSUMN{window}"] = volume_loss / absolute_volume
        output[f"VSUMD{window}"] = (volume_gain - volume_loss) / absolute_volume

    return pd.DataFrame(output, index=frame.index).loc[:, ALPHA158_FEATURES]


def _last_rank(values: np.ndarray) -> float:
    return float((values <= values[-1]).sum() / len(values))


def _rolling_linear_stats(
    series: pd.Series, window: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    values = series.to_numpy(dtype=float)
    if len(values) < window:
        empty = pd.Series(np.nan, index=series.index, dtype=float)
        return empty.copy(), empty.copy(), empty.copy()

    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    x_square_sum = np.square(centered_x).sum()
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    numerator = np.convolve(filled, centered_x[::-1], mode="valid")
    counts = np.convolve(finite.astype(float), np.ones(window), mode="valid")
    rolling_mean = np.convolve(filled, np.ones(window), mode="valid") / window
    rolling_square_mean = (
        np.convolve(np.square(filled), np.ones(window), mode="valid") / window
    )
    y_square_sum = window * (rolling_square_mean - np.square(rolling_mean))
    slope_values = numerator / x_square_sum
    rsquare_values = np.square(numerator) / (x_square_sum * y_square_sum + EPSILON)
    residual_values = (
        values[window - 1 :]
        - rolling_mean
        - slope_values * (x[-1] - x.mean())
    )
    invalid = counts < window
    slope_values[invalid] = np.nan
    rsquare_values[invalid] = np.nan
    residual_values[invalid] = np.nan

    index = series.index
    prefix = np.full(window - 1, np.nan)
    return (
        pd.Series(np.concatenate([prefix, slope_values]), index=index),
        pd.Series(np.concatenate([prefix, rsquare_values]), index=index),
        pd.Series(np.concatenate([prefix, residual_values]), index=index),
    )
