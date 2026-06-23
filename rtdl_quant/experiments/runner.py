from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from rtdl_quant.backtest import (
    GroupBacktest,
    ICAnalysis,
    load_float_market_cap,
    load_industry_map,
    neutralize_cross_sectional_signal,
)
from rtdl_quant.datasets import (
    DatasetSplit,
    add_cross_sectional_rank_label,
    build_dataloaders,
)
from rtdl_quant.metrics import mae, mse, rmse
from rtdl_quant.models import MLP, MLPConfig, ResNet, ResNetConfig
from rtdl_quant.trainer import Trainer, TrainerConfig
from rtdl_quant.utils import save_config, seed_everything

LOGGER = logging.getLogger(__name__)


class _NumericalFTTransformer(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return self.model(x_num, None)


class ExperimentRunner:
    """Run one YAML-defined train/evaluate/backtest experiment."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        experiment = config["experiment"]
        self.output_dir = (
            Path(experiment.get("output_dir", "rtdl_quant/outputs"))
            / experiment["name"]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._configure_logging()

    def _configure_logging(self) -> None:
        log_path = self.output_dir / "train.log"
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path.resolve()
            for handler in root.handlers
        ):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            root.addHandler(handler)

    def run(self) -> dict[str, float]:
        seed = int(self.config["experiment"].get("seed", 42))
        seed_everything(seed)
        save_config(self.config, self.output_dir / "config.yaml")

        frame, feature_columns = self._load_frame()
        loaders = self._build_loaders(frame, feature_columns)
        model, optimizer = self._build_model(len(feature_columns))
        trainer_config = self._trainer_config()
        trainer = Trainer(model, trainer_config, optimizer=optimizer)
        fit_result = trainer.fit(
            loaders["train"], loaders["valid"], checkpoint_name="model.pt"
        )

        predictions = trainer.predict(loaders["test"]).numpy()
        test_dataset = loaders["test"].dataset
        labels = test_dataset.y.numpy()
        future_returns = test_dataset.future_returns
        if future_returns is None:
            LOGGER.warning(
                "No future_return column was found; grouped returns use labels as a proxy"
            )
            future_returns = labels
        evaluation = pd.DataFrame(
            {
                "date": pd.to_datetime(test_dataset.dates),
                "code": test_dataset.codes,
                "raw_prediction": predictions,
                "label": labels,
                "future_return": future_returns,
            }
        )
        evaluation["prediction"] = evaluation["raw_prediction"]
        raw_ic_analysis = ICAnalysis(
            evaluation, prediction_column="raw_prediction"
        )
        raw_daily_ic = raw_ic_analysis.result()
        raw_summary = raw_ic_analysis.summary(raw_daily_ic)
        raw_group_result = GroupBacktest(
            evaluation, prediction_column="raw_prediction"
        ).run()

        neutralization_summary = self._neutralize_predictions(evaluation, frame)
        if neutralization_summary is not None:
            evaluation, coverage = neutralization_summary
            pd.DataFrame([asdict(coverage)]).to_csv(
                self.output_dir / "neutralization_summary.csv", index=False
            )
            raw_daily_ic.to_csv(self.output_dir / "daily_ic_raw.csv")
            raw_group_result.group_returns.to_csv(
                self.output_dir / "group_returns_raw.csv"
            )
        evaluation.to_parquet(self.output_dir / "predictions.parquet", index=False)

        ic_analysis = ICAnalysis(evaluation)
        daily_ic = ic_analysis.result()
        daily_ic.to_csv(self.output_dir / "daily_ic.csv")
        summary = ic_analysis.summary(daily_ic)
        group_result = GroupBacktest(evaluation).run()
        group_result.group_returns.to_csv(self.output_dir / "group_returns.csv")

        metrics = {
            "mse": mse(labels, predictions),
            "rmse": rmse(labels, predictions),
            "mae": mae(labels, predictions),
            "ic": summary.ic_mean,
            "rank_ic": summary.rank_ic_mean,
            "icir": summary.icir,
            "rank_icir": summary.rank_icir,
            "best_validation_loss": fit_result.validation_loss,
            "best_epoch": float(fit_result.best_epoch),
            "top_bottom_mean": float(group_result.top_bottom_spread.mean()),
            "raw_ic": raw_summary.ic_mean,
            "raw_rank_ic": raw_summary.rank_ic_mean,
            "raw_icir": raw_summary.icir,
            "raw_rank_icir": raw_summary.rank_icir,
            "raw_top_bottom_mean": float(
                raw_group_result.top_bottom_spread.mean()
            ),
        }
        pd.DataFrame([metrics]).to_csv(self.output_dir / "metrics.csv", index=False)
        pd.DataFrame(fit_result.history).to_csv(
            self.output_dir / "training_history.csv", index=False
        )
        LOGGER.info("experiment_complete metrics=%s", metrics)
        return metrics

    def _neutralize_predictions(
        self, evaluation: pd.DataFrame, source_frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, Any] | None:
        config = self.config.get("evaluation", {}).get("neutralization", {})
        if not config.get("enabled", False):
            return None

        industry = load_industry_map(
            config.get("industry_path", "industry/industry.csv"),
            code_column=config.get("industry_code_column", "symbol"),
            industry_column=config.get("industry_column", "industry"),
        )
        enriched = evaluation.merge(industry, on="code", how="left")

        if "float_market_cap" in source_frame.columns:
            market_cap = source_frame[
                ["date", "code", "float_market_cap"]
            ].copy()
            market_cap["date"] = pd.to_datetime(market_cap["date"])
        else:
            market_cap = load_float_market_cap(
                self.config["data"].get("prices_dir", "prices"),
                enriched["code"].unique(),
                start_date=enriched["date"].min(),
                end_date=enriched["date"].max(),
            )
        enriched = enriched.merge(
            market_cap, on=["date", "code"], how="left"
        )
        enriched, summary = neutralize_cross_sectional_signal(
            enriched,
            signal_column="raw_prediction",
            output_column="prediction",
            neutralize_industry=bool(config.get("industry", True)),
            neutralize_market_cap=bool(config.get("market_cap", True)),
            standardize=bool(config.get("standardize", True)),
        )
        LOGGER.info("neutralization_summary=%s", asdict(summary))
        return enriched, summary

    def _load_frame(self) -> tuple[pd.DataFrame, list[str]]:
        data_config = self.config["data"]
        source = data_config.get("source", "file").lower()
        path = Path(data_config["path"])
        if source == "prices" and not path.exists():
            if not data_config.get("auto_prepare", False):
                raise FileNotFoundError(
                    f"{path} does not exist. Build it with "
                    "`python -m rtdl_quant.scripts.build_prices_dataset`."
                )
            from rtdl_quant.datasets.prices_alpha158 import (
                PricesAlpha158Builder,
                PricesBuildConfig,
            )

            build = data_config.get("prices_build", {})
            path = PricesAlpha158Builder(
                PricesBuildConfig(
                    prices_dir=data_config.get("prices_dir", "prices"),
                    output_path=path,
                    start_date=build.get("start_date"),
                    end_date=build.get("end_date"),
                    horizon=int(build.get("horizon", 20)),
                    exclude_st=bool(build.get("exclude_st", True)),
                    require_trading=bool(build.get("require_trading", True)),
                    max_instruments=build.get("max_instruments"),
                )
            ).build_to_parquet()
        if path.suffix == ".parquet":
            filters = self._parquet_instrument_filter(
                path, data_config.get("load_max_instruments")
            )
            frame = pd.read_parquet(path, filters=filters)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError("Data path must end in .parquet or .csv")
        label_column = data_config.get("label_column", "label")
        future_return_column = data_config.get(
            "future_return_column", "future_return"
        )
        if label_column not in frame and future_return_column in frame:
            frame = add_cross_sectional_rank_label(
                frame,
                future_return_column=future_return_column,
                date_column=data_config.get("date_column", "date"),
                output_column=label_column,
            )

        explicit_features = data_config.get("feature_columns")
        prefix = data_config.get("feature_prefix", "feature_")
        if explicit_features:
            feature_columns = list(explicit_features)
        elif source == "prices":
            from rtdl_quant.datasets.prices_alpha158 import ALPHA158_FEATURES

            feature_columns = [
                column for column in ALPHA158_FEATURES if column in frame.columns
            ]
        elif prefix:
            feature_columns = [
                column for column in frame.columns if column.startswith(prefix)
            ]
        else:
            feature_columns = []
        if not feature_columns:
            excluded = {
                data_config.get("label_column", "label"),
                data_config.get("future_return_column", "future_return"),
                data_config.get("date_column", "date"),
                data_config.get("code_column", "code"),
            }
            feature_columns = [
                column
                for column in frame.select_dtypes(include=[np.number]).columns
                if column not in excluded
            ]
        if not feature_columns:
            raise ValueError("Could not infer any numerical feature columns")
        return frame, feature_columns

    @staticmethod
    def _parquet_instrument_filter(
        path: Path, max_instruments: int | None
    ) -> list[tuple[str, str, list[str]]] | None:
        """Select an evenly spaced stock subset using Parquet row-group metadata."""
        if max_instruments is None:
            return None
        if max_instruments <= 0:
            raise ValueError("data.load_max_instruments must be positive or null")

        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        code_index = parquet.schema_arrow.get_field_index("code")
        if code_index < 0:
            raise KeyError("Parquet dataset has no 'code' column")

        codes: list[str] = []
        for index in range(parquet.num_row_groups):
            column = parquet.metadata.row_group(index).column(code_index)
            statistics = column.statistics
            if statistics is None or statistics.min != statistics.max:
                codes = (
                    pd.read_parquet(path, columns=["code"])["code"]
                    .drop_duplicates()
                    .sort_values()
                    .astype(str)
                    .tolist()
                )
                break
            value = statistics.min
            codes.append(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
            )
        codes = sorted(set(codes))
        if max_instruments >= len(codes):
            return None
        indices = np.linspace(0, len(codes) - 1, max_instruments, dtype=int)
        selected = [codes[index] for index in indices]
        LOGGER.info(
            "Loading %d of %d instruments from %s",
            len(selected),
            len(codes),
            path,
        )
        return [("code", "in", selected)]

    def _build_loaders(
        self, frame: pd.DataFrame, feature_columns: list[str]
    ) -> dict[str, Any]:
        data = self.config["data"]
        splits = {
            name: DatasetSplit(**bounds) for name, bounds in data["splits"].items()
        }
        return build_dataloaders(
            frame,
            splits,
            batch_size=int(data.get("batch_size", 256)),
            num_workers=int(data.get("num_workers", 0)),
            feature_columns=feature_columns,
            label_column=data.get("label_column", "label"),
            date_column=data.get("date_column", "date"),
            code_column=data.get("code_column", "code"),
            future_return_column=data.get("future_return_column", "future_return"),
        )

    def _build_model(
        self, d_in: int
    ) -> tuple[nn.Module, torch.optim.Optimizer | None]:
        model_config = dict(self.config["model"])
        name = model_config.pop("name").lower()
        model_config["d_in"] = d_in
        if name == "mlp":
            if "hidden_dims" in model_config:
                model_config["hidden_dims"] = tuple(model_config["hidden_dims"])
            return MLP(MLPConfig(**model_config)), None
        if name == "resnet":
            return ResNet(ResNetConfig(**model_config)), None
        if name in {"fttransformer", "ft_transformer"}:
            try:
                import rtdl_revisiting_models as rtdl
            except ImportError as error:
                raise ImportError(
                    "FT-Transformer requires rtdl-revisiting-models"
                ) from error
            model_config.pop("d_in")
            n_blocks = int(model_config.pop("n_blocks", 2))
            if hasattr(rtdl.FTTransformer, "make_default"):
                official = rtdl.FTTransformer.make_default(
                    n_num_features=d_in,
                    cat_cardinalities=None,
                    d_out=1,
                    n_blocks=n_blocks,
                    **model_config,
                )
            else:
                backbone_config = rtdl.FTTransformer.get_default_kwargs(n_blocks)
                backbone_config.update(model_config)
                backbone_config["d_out"] = 1
                official = rtdl.FTTransformer(
                    n_cont_features=d_in,
                    cat_cardinalities=[],
                    **backbone_config,
                )
            wrapped = _NumericalFTTransformer(official)
            trainer_config = self.config["trainer"]
            if hasattr(official, "make_parameter_groups"):
                parameter_groups = official.make_parameter_groups()
            else:
                parameter_groups = rtdl.get_parameter_groups(official)
            optimizer = AdamW(
                parameter_groups,
                lr=float(trainer_config.get("learning_rate", 1e-4)),
                weight_decay=float(trainer_config.get("weight_decay", 1e-5)),
            )
            return wrapped, optimizer
        raise ValueError(f"Unsupported model: {name}")

    def _trainer_config(self) -> TrainerConfig:
        values = dict(self.config["trainer"])
        values["checkpoint_dir"] = str(self.output_dir)
        values.setdefault("tensorboard_dir", str(self.output_dir / "tensorboard"))
        return TrainerConfig(**values)
