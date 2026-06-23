import numpy as np
import pandas as pd

from rtdl_quant.backtest import GroupBacktest, ICAnalysis


def make_panel() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2020-01-01", periods=3):
        for index in range(10):
            rows.append(
                {
                    "date": date,
                    "code": f"S{index}",
                    "prediction": float(index),
                    "label": float(index),
                    "future_return": index / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_ic_analysis() -> None:
    result = ICAnalysis(make_panel()).result()
    assert np.allclose(result["ic"], 1.0)
    assert np.allclose(result["rank_ic"], 1.0)
    assert np.isclose(result["cumulative_ic"].iloc[-1], 3.0)


def test_group_backtest_top_minus_bottom() -> None:
    result = GroupBacktest(make_panel(), n_groups=5).run()
    assert list(result.group_returns.columns) == [
        "group_1",
        "group_2",
        "group_3",
        "group_4",
        "group_5",
    ]
    assert (result.top_bottom_spread > 0).all()
