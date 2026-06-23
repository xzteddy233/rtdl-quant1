import numpy as np
import pandas as pd
import pytest

from rtdl_quant.datasets import (
    Alpha158Dataset,
    DatasetSplit,
    add_cross_sectional_rank_label,
    build_dataloaders,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "code": ["A", "B", "C", "D"],
            "f0": [1.0, 2.0, 3.0, 4.0],
            "f1": [4.0, 3.0, 2.0, 1.0],
            "label": [0.1, 0.2, 0.3, 0.4],
            "future_return": [0.01, 0.02, 0.03, 0.04],
        }
    )


def test_dataset_split_and_item_contract() -> None:
    dataset = Alpha158Dataset(
        make_frame(),
        feature_columns=["f0", "f1"],
        split=DatasetSplit("2020-01-02", "2020-01-03"),
    )
    assert len(dataset) == 2
    item = dataset[0]
    assert set(item) == {"x_num", "y", "date", "code"}
    assert item["x_num"].shape == (2,)
    assert item["date"] == "2020-01-02"
    assert np.isclose(dataset.future_returns[0], 0.02)


def test_dataloader_collates_metadata() -> None:
    frame = make_frame()
    loaders = build_dataloaders(
        frame,
        {
            "train": DatasetSplit("2020-01-01", "2020-01-02"),
            "valid": DatasetSplit("2020-01-03", "2020-01-03"),
            "test": DatasetSplit("2020-01-04", "2020-01-04"),
        },
        feature_columns=["f0", "f1"],
        batch_size=2,
    )
    batch = next(iter(loaders["train"]))
    assert batch["x_num"].shape == (2, 2)
    assert len(batch["date"]) == 2


def test_cross_sectional_rank_label() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "future_return": [-0.1, 0.0, 0.2],
        }
    )
    ranked = add_cross_sectional_rank_label(frame)
    assert np.allclose(ranked["label"], [1 / 3, 2 / 3, 1.0])


def test_overlapping_splits_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-overlapping"):
        build_dataloaders(
            make_frame(),
            {
                "train": DatasetSplit("2020-01-01", "2020-01-03"),
                "valid": DatasetSplit("2020-01-03", "2020-01-03"),
                "test": DatasetSplit("2020-01-04", "2020-01-04"),
            },
            feature_columns=["f0", "f1"],
        )
