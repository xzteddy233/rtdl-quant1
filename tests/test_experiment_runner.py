from pathlib import Path

import numpy as np
import pandas as pd

from rtdl_quant.experiments.runner import ExperimentRunner


def test_parquet_instrument_filter_uses_row_group_metadata(tmp_path: Path) -> None:
    path = tmp_path / "panel.parquet"
    writer = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for code in ["SH600000", "SH600001", "SZ000001", "SZ000002"]:
            table = pa.Table.from_pandas(
                pd.DataFrame(
                    {
                        "date": pd.date_range("2024-01-01", periods=2),
                        "code": [code, code],
                        "feature": [1.0, 2.0],
                    }
                ),
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    filters = ExperimentRunner._parquet_instrument_filter(path, 2)
    assert filters is not None
    selected = filters[0][2]
    assert selected == ["SH600000", "SZ000002"]
    loaded = pd.read_parquet(path, filters=filters)
    assert sorted(loaded["code"].unique()) == selected


def test_parquet_instrument_filter_can_load_full_universe(tmp_path: Path) -> None:
    path = tmp_path / "panel.parquet"
    pd.DataFrame({"code": ["A", "B"], "value": [1.0, 2.0]}).to_parquet(path)
    assert ExperimentRunner._parquet_instrument_filter(path, None) is None


def test_tree_frame_splits_drop_invalid_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2021-01-01",
                    "2022-01-01",
                ]
            ),
            "code": ["A", "B", "A", "A"],
            "f0": [1.0, float("nan"), 3.0, 4.0],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "label": [0.1, 0.2, 0.3, 0.4],
            "future_return": [0.01, 0.02, 0.03, 0.04],
        }
    )
    runner = ExperimentRunner(
        {
            "experiment": {
                "name": "test_tree_split",
                "seed": 1,
                "output_dir": str(tmp_path),
            },
            "data": {
                "splits": {
                    "train": {"start": "2020-01-01", "end": "2020-12-31"},
                    "valid": {"start": "2021-01-01", "end": "2021-12-31"},
                    "test": {"start": "2022-01-01", "end": "2022-12-31"},
                }
            },
            "model": {"name": "xgboost"},
        }
    )
    splits = runner._build_frame_splits(frame, ["f0", "f1"])
    assert splits["train"]["x"].shape == (1, 2)
    assert np.allclose(splits["valid"]["y"], [0.3])
    assert np.allclose(splits["test"]["future_returns"], [0.04])
