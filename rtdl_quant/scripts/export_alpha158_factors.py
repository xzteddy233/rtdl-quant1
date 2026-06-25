from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rtdl_quant.datasets.prices_alpha158 import (
    PricesAlpha158Builder,
    PricesBuildConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Alpha158 factors from prices/*.csv without training"
    )
    parser.add_argument("--prices-dir", type=Path, default=Path("prices"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/alpha158_factors.parquet")
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-instruments", type=int)
    parser.add_argument("--include-st", action="store_true")
    parser.add_argument("--include-suspended", action="store_true")
    parser.add_argument(
        "--include-future-return",
        action="store_true",
        help="Also export raw future horizon return; final horizon rows are dropped.",
    )
    parser.add_argument(
        "--include-market-cap",
        action="store_true",
        help="Also export estimated daily free-float market cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    builder = PricesAlpha158Builder(
        PricesBuildConfig(
            prices_dir=args.prices_dir,
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            horizon=args.horizon,
            exclude_st=not args.include_st,
            require_trading=not args.include_suspended,
            max_instruments=args.max_instruments,
        )
    )
    output = builder.build_factors_to_parquet(
        include_future_return=args.include_future_return,
        include_market_cap=args.include_market_cap,
    )
    print(f"Alpha158 factors saved to {output}")


if __name__ == "__main__":
    main()
