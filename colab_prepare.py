from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare RTDL Quant for Google Colab: copy data from Drive, "
            "check float_market_cap, set universe size, and configure neutralization."
        )
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive"),
        help="Google Drive root path mounted in Colab.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help=(
            "Path to alpha158_prices.parquet. Defaults to "
            "<drive-root>/alpha158_prices.parquet."
        ),
    )
    parser.add_argument(
        "--industry",
        type=Path,
        default=None,
        help="Path to industry.csv. Defaults to <drive-root>/industry.csv.",
    )
    parser.add_argument(
        "--load-max-instruments",
        type=int,
        default=500,
        help="Number of stocks to load for each training run. Use 500 for normal Colab.",
    )
    parser.add_argument(
        "--neutralization",
        choices=("auto", "full", "industry", "off"),
        default="auto",
        help=(
            "auto: full if industry.csv and float_market_cap exist, industry-only "
            "if only industry.csv exists, otherwise off."
        ),
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Do not copy files from Drive; only inspect and update configs.",
    )
    return parser.parse_args()


def copy_if_available(source: Path, target: Path, *, required: bool) -> bool:
    if not source.exists():
        message = f"missing: {source}"
        if required:
            raise FileNotFoundError(message)
        print(message)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"copying {source} -> {target}")
    shutil.copy2(source, target)
    print(f"copied: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
    return True


def parquet_has_column(path: Path, column: str) -> bool:
    schema = pq.ParquetFile(path).schema_arrow
    return column in schema.names


def update_nested(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = config
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def configure_yaml_files(
    *,
    config_dir: Path,
    load_max_instruments: int,
    neutralization_mode: str,
) -> None:
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No YAML configs found under {config_dir}")

    for path in paths:
        config = yaml.safe_load(path.read_text()) or {}
        update_nested(config, ("data", "load_max_instruments"), load_max_instruments)

        neutralization = config.setdefault("evaluation", {}).setdefault(
            "neutralization", {}
        )
        if neutralization_mode == "off":
            neutralization["enabled"] = False
        elif neutralization_mode == "industry":
            neutralization["enabled"] = True
            neutralization["industry"] = True
            neutralization["market_cap"] = False
            neutralization["standardize"] = True
        elif neutralization_mode == "full":
            neutralization["enabled"] = True
            neutralization["industry"] = True
            neutralization["market_cap"] = True
            neutralization["standardize"] = True
        else:
            raise ValueError(f"Unexpected neutralization mode: {neutralization_mode}")

        path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(
            f"updated {path}: load_max_instruments={load_max_instruments}, "
            f"neutralization={neutralization_mode}"
        )


def main() -> None:
    args = parse_args()
    parquet_source = args.parquet or args.drive_root / "alpha158_prices.parquet"
    industry_source = args.industry or args.drive_root / "industry.csv"
    local_parquet = Path("data/alpha158_prices.parquet")
    local_industry = Path("industry/industry.csv")

    if not args.skip_copy:
        copy_if_available(parquet_source, local_parquet, required=True)
        copy_if_available(industry_source, local_industry, required=False)

    if not local_parquet.exists():
        raise FileNotFoundError(
            f"{local_parquet} does not exist. Copy alpha158_prices.parquet first."
        )

    has_market_cap = parquet_has_column(local_parquet, "float_market_cap")
    has_industry = local_industry.exists()

    print(f"has float_market_cap: {has_market_cap}")
    print(f"has industry.csv: {has_industry}")

    if args.neutralization == "auto":
        if has_industry and has_market_cap:
            neutralization_mode = "full"
        elif has_industry:
            neutralization_mode = "industry"
        else:
            neutralization_mode = "off"
    else:
        neutralization_mode = args.neutralization

    if neutralization_mode == "full" and not has_market_cap:
        raise ValueError(
            "Requested full neutralization, but alpha158_prices.parquet has no "
            "float_market_cap column. Use --neutralization industry/off or rebuild the cache."
        )
    if neutralization_mode in {"full", "industry"} and not has_industry:
        raise ValueError(
            "Requested industry neutralization, but industry/industry.csv is missing. "
            "Upload industry.csv or use --neutralization off."
        )

    configure_yaml_files(
        config_dir=Path("rtdl_quant/configs"),
        load_max_instruments=args.load_max_instruments,
        neutralization_mode=neutralization_mode,
    )

    print("\nReady. Suggested training commands:")
    print("python -m rtdl_quant.main --config rtdl_quant/configs/mlp.yaml")
    print("python -m rtdl_quant.main --config rtdl_quant/configs/resnet.yaml")
    print("python -m rtdl_quant.main --config rtdl_quant/configs/ft_transformer.yaml")
    print("python -m rtdl_quant.main --config rtdl_quant/configs/catboost.yaml")
    print("python -m rtdl_quant.main --config rtdl_quant/configs/xgboost.yaml")


if __name__ == "__main__":
    main()
