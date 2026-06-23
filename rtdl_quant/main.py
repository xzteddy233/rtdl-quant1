from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtdl_quant.experiments import ExperimentRunner
from rtdl_quant.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an RTDL quant experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "config.yaml",
        help="Path to an experiment YAML file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = ExperimentRunner(load_config(args.config)).run()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
