from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SplitName = Literal["train", "valid", "test"]


def add_cross_sectional_rank_label(
    frame: pd.DataFrame,
    *,
    future_return_column: str = "future_return",
    date_column: str = "date",
    output_column: str = "label",
) -> pd.DataFrame:
    """Add a within-date percentile-rank target while retaining raw returns."""
    if future_return_column not in frame or date_column not in frame:
        raise KeyError(f"Expected '{future_return_column}' and '{date_column}' columns")
    result = frame.copy()
    result[output_column] = result.groupby(date_column)[future_return_column].rank(
        method="average", pct=True
    )
    return result


@dataclass(frozen=True)
class DatasetSplit:
    """Inclusive date boundaries for one chronological dataset split."""

    start: str | pd.Timestamp
    end: str | pd.Timestamp

    def mask(self, dates: pd.Series) -> pd.Series:
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if start > end:
            raise ValueError(f"Split start {start} is after end {end}")
        return dates.between(start, end, inclusive="both")


class Alpha158Dataset(Dataset[dict[str, Any]]):
    """PyTorch dataset for cross-sectional Alpha158 observations.

    The canonical frame has one row per ``(date, code)``, 158 numeric factor
    columns, and one label column containing a future-return target (typically
    the within-date rank of the next 20 trading-day return).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str] | None = None,
        label_column: str = "label",
        future_return_column: str | None = "future_return",
        date_column: str = "date",
        code_column: str = "code",
        split: DatasetSplit | None = None,
        dtype: torch.dtype = torch.float32,
        drop_invalid: bool = True,
    ) -> None:
        data = self._normalize_frame(frame, date_column, code_column)
        if split is not None:
            data = data.loc[split.mask(data[date_column])]

        if feature_columns is None:
            excluded = {label_column, date_column, code_column}
            feature_columns = [c for c in data.columns if c not in excluded]
        self.feature_columns = tuple(feature_columns)
        if not self.feature_columns:
            raise ValueError("No feature columns were provided or inferred")

        required = {*self.feature_columns, label_column, date_column, code_column}
        missing = required.difference(data.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        numeric = data.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
        labels = pd.to_numeric(data[label_column], errors="coerce")
        valid = np.isfinite(numeric.to_numpy()).all(axis=1) & np.isfinite(labels.to_numpy())
        if not valid.all():
            if not drop_invalid:
                raise ValueError(f"Found {(~valid).sum()} rows with NaN or infinite values")
            data = data.loc[valid].copy()
            numeric = numeric.loc[valid]
            labels = labels.loc[valid]

        self.x_num = torch.as_tensor(numeric.to_numpy(dtype=np.float32), dtype=dtype)
        self.y = torch.as_tensor(labels.to_numpy(dtype=np.float32), dtype=dtype)
        self.dates = data[date_column].dt.strftime("%Y-%m-%d").to_numpy(copy=True)
        self.codes = data[code_column].astype(str).to_numpy(copy=True)
        if future_return_column and future_return_column in data.columns:
            future_returns = pd.to_numeric(
                data[future_return_column], errors="coerce"
            ).to_numpy(dtype=np.float32)
            self.future_returns: np.ndarray | None = future_returns
        else:
            self.future_returns = None

    @staticmethod
    def _normalize_frame(
        frame: pd.DataFrame, date_column: str, code_column: str
    ) -> pd.DataFrame:
        data = frame.copy()
        if isinstance(data.index, pd.MultiIndex):
            index_names = set(data.index.names)
            if {date_column, code_column}.issubset(index_names):
                data = data.reset_index()
        if date_column not in data.columns or code_column not in data.columns:
            raise KeyError(
                f"Expected '{date_column}' and '{code_column}' columns, or a named MultiIndex"
            )
        data[date_column] = pd.to_datetime(data[date_column])
        return data.sort_values([date_column, code_column]).reset_index(drop=True)

    @classmethod
    def from_qlib(
        cls,
        handler: Any,
        *,
        segment: str | slice,
        feature_group: str = "feature",
        label_group: str = "label",
        label_column: str = "label",
        future_return_column: str = "future_return",
        rank_label: bool = True,
        **kwargs: Any,
    ) -> "Alpha158Dataset":
        """Create a dataset from a fitted Qlib ``DataHandlerLP``-like object.

        Qlib is deliberately imported by the caller, so the core package stays
        usable without installing the comparatively heavy ``pyqlib`` extra.
        """
        frame = handler.fetch(
            selector=segment,
            col_set=[feature_group, label_group],
            data_key=getattr(handler, "DK_L", "learn"),
        )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("handler.fetch(...) must return a pandas DataFrame")

        if isinstance(frame.columns, pd.MultiIndex):
            features = frame[feature_group].copy()
            label_frame = frame[label_group]
            features.columns = [str(column) for column in features.columns]
            raw_return = label_frame.iloc[:, 0]
            features[future_return_column] = raw_return
            if rank_label:
                date_level = (
                    "datetime"
                    if "datetime" in raw_return.index.names
                    else raw_return.index.names[0]
                )
                features[label_column] = raw_return.groupby(level=date_level).rank(
                    method="average", pct=True
                )
            else:
                features[label_column] = raw_return
            frame = features

        if isinstance(frame.index, pd.MultiIndex):
            names = list(frame.index.names)
            rename = {}
            if "datetime" in names:
                rename["datetime"] = "date"
            if "instrument" in names:
                rename["instrument"] = "code"
            frame = frame.rename_axis(index=rename).reset_index()
        return cls(
            frame,
            label_column=label_column,
            future_return_column=future_return_column,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "x_num": self.x_num[index],
            "y": self.y[index],
            "date": self.dates[index],
            "code": self.codes[index],
        }


def build_dataloaders(
    frame: pd.DataFrame,
    splits: Mapping[SplitName, DatasetSplit],
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    feature_columns: Sequence[str] | None = None,
    **dataset_kwargs: Any,
) -> dict[SplitName, DataLoader[dict[str, Any]]]:
    """Build chronological train/validation/test loaders from one frame."""
    required_splits = {"train", "valid", "test"}
    if set(splits) != required_splits:
        raise ValueError(f"splits must contain exactly {sorted(required_splits)}")
    ordered = [splits["train"], splits["valid"], splits["test"]]
    for left, right in zip(ordered, ordered[1:]):
        if pd.Timestamp(left.end) >= pd.Timestamp(right.start):
            raise ValueError("train/valid/test date ranges must be ordered and non-overlapping")

    loaders: dict[SplitName, DataLoader[dict[str, Any]]] = {}
    for name in ("train", "valid", "test"):
        dataset = Alpha158Dataset(
            frame,
            split=splits[name],
            feature_columns=feature_columns,
            **dataset_kwargs,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders


if __name__ == "__main__":
    # Minimal example; replace ``frame`` with normalized Qlib Alpha158 output.
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    frame = pd.DataFrame(rng.normal(size=(300, 158)), columns=[f"f{i}" for i in range(158)])
    frame["label"] = rng.normal(size=len(frame))
    frame["date"] = np.repeat(dates, 10)
    frame["code"] = [f"stock_{i % 10:03d}" for i in range(len(frame))]
    loaders = build_dataloaders(
        frame,
        {
            "train": DatasetSplit("2020-01-01", "2020-01-28"),
            "valid": DatasetSplit("2020-01-29", "2020-02-05"),
            "test": DatasetSplit("2020-02-06", "2020-02-11"),
        },
        batch_size=32,
    )
    print(next(iter(loaders["train"])))
