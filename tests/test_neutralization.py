from pathlib import Path

import numpy as np
import pandas as pd

from rtdl_quant.backtest.neutralization import (
    load_float_market_cap,
    load_industry_map,
    neutralize_cross_sectional_signal,
)


def test_load_float_market_cap_uses_close_volume_and_turnover(
    tmp_path: Path,
) -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["SH600000", "SH600000"],
            "close": [10.0, 12.0],
            "volume": [1_000.0, 2_000.0],
            "turn": [2.0, 4.0],
        }
    )
    prices.to_csv(tmp_path / "SH600000.csv", index=False, encoding="utf-8-sig")
    result = load_float_market_cap(tmp_path, ["SH600000"])
    assert np.allclose(result["float_market_cap"], [500_000.0, 600_000.0])


def test_load_industry_map_normalizes_codes(tmp_path: Path) -> None:
    path = tmp_path / "industry.csv"
    pd.DataFrame(
        {"symbol": ["sh600000"], "industry": ["Bank"]}
    ).to_csv(path, index=False, encoding="utf-8-sig")
    result = load_industry_map(path)
    assert result.to_dict("records") == [
        {"code": "SH600000", "industry": "Bank"}
    ]


def test_neutralization_removes_industry_and_size_exposure() -> None:
    rows = []
    for industry_index, industry in enumerate(["A", "B"]):
        for stock_index in range(10):
            log_cap = 8.0 + stock_index / 10.0
            alpha = (-1.0) ** stock_index * (stock_index + 1) / 10.0
            rows.append(
                {
                    "date": "2024-01-02",
                    "code": f"{industry}{stock_index}",
                    "industry": industry,
                    "float_market_cap": np.exp(log_cap),
                    "raw_prediction": (
                        industry_index * 5.0 + 2.0 * log_cap + alpha
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    result, summary = neutralize_cross_sectional_signal(frame)
    assert summary.valid_rows == len(frame)
    industry_means = result.groupby("industry")["prediction"].mean()
    assert np.allclose(industry_means, 0.0, atol=1e-12)
    assert abs(
        np.corrcoef(
            result["prediction"], np.log(result["float_market_cap"])
        )[0, 1]
    ) < 1e-12
