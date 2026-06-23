from pathlib import Path

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
