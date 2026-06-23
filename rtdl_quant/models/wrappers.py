from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

try:
    import rtdl_revisiting_models as rtdl
except ImportError:
    rtdl = None


class BaseModel(ABC):
    """Backend-neutral interface for neural and tree-based estimators."""

    @abstractmethod
    def fit(
        self,
        x_num: ArrayLike,
        y: ArrayLike,
        *,
        x_valid: ArrayLike | None = None,
        y_valid: ArrayLike | None = None,
    ) -> "BaseModel": ...

    @abstractmethod
    def predict(self, x_num: ArrayLike) -> NDArray[np.float32]: ...

    @abstractmethod
    def save(self, path: str | Path) -> Path: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs: Any) -> Self: ...


class TorchModel(BaseModel):
    """Small sklearn-like adapter around an RTDL PyTorch module."""

    def __init__(
        self,
        module: nn.Module,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 12,
        device: str = "auto",
        optimizer_parameter_groups: list[dict[str, Any]] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.module = module.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model_config = model_config or {}
        parameters = optimizer_parameter_groups or [{"params": self.module.parameters()}]
        self.optimizer = AdamW(
            parameters, lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def fit(
        self,
        x_num: ArrayLike,
        y: ArrayLike,
        *,
        x_valid: ArrayLike | None = None,
        y_valid: ArrayLike | None = None,
    ) -> "TorchModel":
        x = _as_float_tensor(x_num)
        target = _as_float_tensor(y).view(-1)
        if x_valid is None or y_valid is None:
            x_valid, y_valid = x_num, y
        valid_x = _as_float_tensor(x_valid)
        valid_y = _as_float_tensor(y_valid).view(-1)
        loader = DataLoader(
            TensorDataset(x, target), batch_size=self.batch_size, shuffle=True
        )
        loss_fn = nn.MSELoss()
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        stale = 0

        for _ in range(self.epochs):
            self.module.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.module(batch_x).view(-1), batch_y)
                loss.backward()
                self.optimizer.step()

            self.module.eval()
            with torch.inference_mode():
                prediction = self.module(valid_x.to(self.device)).view(-1)
                valid_loss = loss_fn(prediction, valid_y.to(self.device)).item()
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.module.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.module.load_state_dict(best_state)
        return self

    @torch.inference_mode()
    def predict(self, x_num: ArrayLike) -> NDArray[np.float32]:
        self.module.eval()
        tensor = _as_float_tensor(x_num)
        outputs = []
        for (batch,) in DataLoader(TensorDataset(tensor), batch_size=self.batch_size):
            outputs.append(self.module(batch.to(self.device)).view(-1).cpu())
        if not outputs:
            return np.empty(0, dtype=np.float32)
        return torch.cat(outputs).numpy().astype(np.float32, copy=False)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.module.state_dict(),
                "model_config": self.model_config,
            },
            output,
        )
        return output.resolve()

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> Self:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        constructor_config = dict(checkpoint.get("model_config", {}))
        constructor_config.pop("name", None)
        constructor_config.update(kwargs)
        model = cls(**constructor_config)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        return model


class MLPModel(TorchModel):
    def __init__(
        self,
        d_in: int = 158,
        d_layers: tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        package = _require_rtdl()
        if hasattr(package.MLP, "make_baseline"):
            module = package.MLP.make_baseline(
                d_in=d_in, d_layers=list(d_layers), dropout=dropout, d_out=1
            )
        else:
            if len(set(d_layers)) != 1:
                raise ValueError(
                    "rtdl-revisiting-models 0.0.2 requires equal MLP block widths"
                )
            module = package.MLP(
                d_in=d_in,
                d_out=1,
                n_blocks=len(d_layers),
                d_block=d_layers[0],
                dropout=dropout,
            )
        super().__init__(
            module,
            model_config={
                "name": "mlp",
                "d_in": d_in,
                "d_layers": list(d_layers),
                "dropout": dropout,
            },
            **kwargs,
        )


class ResNetModel(TorchModel):
    def __init__(
        self,
        d_in: int = 158,
        n_blocks: int = 4,
        d_block: int = 256,
        d_hidden: int | None = None,
        d_hidden_multiplier: float | None = 2.0,
        dropout1: float = 0.25,
        dropout2: float = 0.0,
        **kwargs: Any,
    ) -> None:
        package = _require_rtdl()
        constructor = getattr(package.ResNet, "make_baseline", package.ResNet)
        module = constructor(
            d_in=d_in,
            d_out=1,
            n_blocks=n_blocks,
            d_block=d_block,
            d_hidden=d_hidden,
            d_hidden_multiplier=d_hidden_multiplier,
            dropout1=dropout1,
            dropout2=dropout2,
        )
        super().__init__(
            module,
            model_config={
                "name": "resnet",
                "d_in": d_in,
                "n_blocks": n_blocks,
                "d_block": d_block,
                "d_hidden": d_hidden,
                "d_hidden_multiplier": d_hidden_multiplier,
                "dropout1": dropout1,
                "dropout2": dropout2,
            },
            **kwargs,
        )


class FTTransformerModel(TorchModel):
    def __init__(
        self,
        n_num_features: int = 158,
        cat_cardinalities: list[int] | None = None,
        n_blocks: int = 2,
        d_block: int | None = None,
        **kwargs: Any,
    ) -> None:
        package = _require_rtdl()
        if cat_cardinalities:
            raise NotImplementedError(
                "The unified x_num-only API currently supports numerical features only"
            )
        if hasattr(package.FTTransformer, "make_default"):
            module = package.FTTransformer.make_default(
                n_num_features=n_num_features,
                cat_cardinalities=None,
                d_out=1,
                n_blocks=n_blocks,
            )
        else:
            backbone_config = package.FTTransformer.get_default_kwargs(n_blocks)
            if d_block is not None:
                backbone_config["d_block"] = d_block
            backbone_config["d_out"] = 1
            module = package.FTTransformer(
                n_cont_features=n_num_features,
                cat_cardinalities=[],
                **backbone_config,
            )

        class NumericalFTTransformer(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x_num: torch.Tensor) -> torch.Tensor:
                return self.model(x_num, None)

        wrapped = NumericalFTTransformer(module)
        parameter_groups = (
            module.make_parameter_groups()
            if hasattr(module, "make_parameter_groups")
            else package.get_parameter_groups(module)
        )
        super().__init__(
            wrapped,
            optimizer_parameter_groups=parameter_groups,
            model_config={
                "name": "ft_transformer",
                "n_num_features": n_num_features,
                "n_blocks": n_blocks,
                "d_block": d_block,
            },
            **kwargs,
        )


def _require_rtdl() -> Any:
    if rtdl is None:
        raise ImportError(
            "Install rtdl-revisiting-models to use the official model wrappers"
        )
    return rtdl


def _as_float_tensor(values: ArrayLike) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    return torch.from_numpy(array)


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
