import numpy as np
import pandas as pd

from rtdl_quant.datasets.prices_alpha158 import (
    ALPHA158_FEATURES,
    PricesAlpha158Builder,
    PricesBuildConfig,
    compute_alpha158,
)


def make_prices(rows: int = 100) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 10.0 + index * 0.03 + np.sin(index / 4.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "code": "sh.600000",
            "symbol": "SH600000",
            "open": close * 0.995,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "preclose": np.r_[close[0], close[:-1]],
            "volume": 1_000_000 + index * 1000 + 100_000 * np.cos(index / 3.0),
            "amount": close * 1_000_000,
            "adjustflag": 2,
            "turn": 1.0,
            "tradestatus": 1,
            "pctChg": 0.0,
            "peTTM": 10.0,
            "pbMRQ": 1.0,
            "psTTM": 2.0,
            "pcfNcfTTM": 5.0,
            "isST": 0,
        }
    )


def test_alpha158_feature_contract() -> None:
    factors = compute_alpha158(make_prices())
    assert factors.shape == (100, 158)
    assert tuple(factors.columns) == ALPHA158_FEATURES
    assert np.isfinite(factors.iloc[-1]).all()


def test_features_do_not_use_future_rows() -> None:
    original = make_prices()
    changed = original.copy()
    changed.loc[80:, "close"] *= 3.0
    before = compute_alpha158(original)
    after = compute_alpha158(changed)
    pd.testing.assert_series_equal(before.loc[79], after.loc[79])


def test_short_price_history_returns_empty_mature_factor_set(tmp_path) -> None:
    path = tmp_path / "SH600001.csv"
    make_prices(rows=30).to_csv(path, index=False, encoding="utf-8-sig")
    builder = PricesAlpha158Builder(
        PricesBuildConfig(prices_dir=tmp_path, horizon=20)
    )
    result = builder.transform_file(path)
    assert result.empty


def test_builder_creates_future_return_and_metadata(tmp_path) -> None:
    path = tmp_path / "SH600000.csv"
    make_prices().to_csv(path, index=False, encoding="utf-8-sig")
    builder = PricesAlpha158Builder(
        PricesBuildConfig(prices_dir=tmp_path, horizon=20)
    )
    result = builder.transform_file(path)
    assert {
        "date",
        "code",
        "future_return",
        "float_market_cap",
    }.issubset(result.columns)
    assert result["code"].eq("SH600000").all()
    assert len(result) > 0


def test_factor_export_does_not_require_future_return(tmp_path) -> None:
    path = tmp_path / "SH600000.csv"
    make_prices(rows=100).to_csv(path, index=False, encoding="utf-8-sig")
    builder = PricesAlpha158Builder(
        PricesBuildConfig(prices_dir=tmp_path, horizon=20)
    )
    factors = builder.transform_factor_file(path)
    dataset = builder.transform_file(path)
    assert set(ALPHA158_FEATURES).issubset(factors.columns)
    assert "future_return" not in factors.columns
    assert len(factors) > len(dataset)


def test_factor_export_can_include_optional_columns(tmp_path) -> None:
    path = tmp_path / "SH600000.csv"
    make_prices(rows=100).to_csv(path, index=False, encoding="utf-8-sig")
    builder = PricesAlpha158Builder(
        PricesBuildConfig(prices_dir=tmp_path, horizon=20)
    )
    result = builder.transform_factor_file(
        path,
        include_future_return=True,
        include_market_cap=True,
    )
    assert {"future_return", "float_market_cap"}.issubset(result.columns)
    assert len(result) > 0
