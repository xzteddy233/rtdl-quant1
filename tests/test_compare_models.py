from pathlib import Path

import pandas as pd

from rtdl_quant.scripts.compare_models import build_comparison


def test_build_comparison_sorts_by_rank_ic(tmp_path: Path) -> None:
    values = {
        "alpha158_mlp": 0.05,
        "alpha158_resnet": 0.15,
        "alpha158_ft_transformer": 0.10,
    }
    for directory, rank_ic in values.items():
        output = tmp_path / directory
        output.mkdir()
        pd.DataFrame(
            [{"mse": 0.1, "ic": rank_ic, "rank_ic": rank_ic}]
        ).to_csv(output / "metrics.csv", index=False)

    comparison = build_comparison(tmp_path)
    assert comparison["model"].tolist() == [
        "resnet",
        "ft_transformer",
        "mlp",
    ]
    assert (tmp_path / "model_comparison.csv").exists()
