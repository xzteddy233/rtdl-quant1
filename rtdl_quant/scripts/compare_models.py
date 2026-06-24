from __future__ import annotations

from pathlib import Path

import pandas as pd

MODEL_OUTPUTS = {
    "mlp": "alpha158_mlp",
    "resnet": "alpha158_resnet",
    "ft_transformer": "alpha158_ft_transformer",
    "xgboost": "alpha158_xgboost",
    "catboost": "alpha158_catboost",
}


def build_comparison(output_root: str | Path = "rtdl_quant/outputs") -> pd.DataFrame:
    root = Path(output_root)
    rows: list[pd.DataFrame] = []
    for model, directory in MODEL_OUTPUTS.items():
        path = root / directory / "metrics.csv"
        if not path.exists():
            continue
        metrics = pd.read_csv(path)
        if len(metrics) != 1:
            raise ValueError(f"Expected one metrics row in {path}")
        metrics.insert(0, "model", model)
        rows.append(metrics)
    if not rows:
        raise FileNotFoundError("No completed model metrics were found")
    comparison = pd.concat(rows, ignore_index=True)
    comparison = comparison.sort_values("rank_ic", ascending=False).reset_index(
        drop=True
    )
    output = root / "model_comparison.csv"
    comparison.to_csv(output, index=False)
    return comparison


def main() -> None:
    comparison = build_comparison()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
