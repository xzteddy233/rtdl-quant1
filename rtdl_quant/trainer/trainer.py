from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # TensorBoard is useful but not required for inference.
    SummaryWriter = None  # type: ignore[assignment,misc]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12
    min_delta: float = 0.0
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = True
    device: str = "auto"
    checkpoint_dir: str = "rtdl_quant/outputs/checkpoints"
    tensorboard_dir: str | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.patience <= 0:
            raise ValueError("epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer hyperparameters")


@dataclass(frozen=True)
class FitResult:
    train_loss: float
    validation_loss: float
    best_model_path: Path
    best_epoch: int
    history: tuple[dict[str, float], ...]


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Extensible supervised regression trainer with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        optimizer: Optimizer | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer = optimizer or AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.amp_enabled = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.writer = self._make_writer(config.tensorboard_dir)

    @staticmethod
    def _make_writer(path: str | None) -> Any:
        if path is None:
            return None
        if SummaryWriter is None:
            LOGGER.warning("TensorBoard is not installed; logging is disabled")
            return None
        return SummaryWriter(log_dir=path)

    @staticmethod
    def _extract_batch(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        return batch["x_num"], batch["y"]

    def train_epoch(self, loader: DataLoader[Any], epoch: int = 0) -> float:
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(
            loader,
            desc=f"train {epoch:03d}",
            leave=False,
            disable=not self.config.show_progress,
        )
        for batch in progress:
            x_num, y = self._extract_batch(batch)
            x_num = x_num.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).float().view(-1)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                prediction = self.model(x_num).view(-1)
                loss = self.loss_fn(prediction, y)

            self.scaler.scale(loss).backward()
            if self.config.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = len(y)
            total_loss += loss.detach().item() * batch_size
            total_examples += batch_size
            progress.set_postfix(loss=f"{loss.detach().item():.6f}")
        if total_examples == 0:
            raise ValueError("Training DataLoader is empty")
        return total_loss / total_examples

    @torch.inference_mode()
    def validate_epoch(self, loader: DataLoader[Any], epoch: int = 0) -> float:
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(
            loader,
            desc=f"valid {epoch:03d}",
            leave=False,
            disable=not self.config.show_progress,
        )
        for batch in progress:
            x_num, y = self._extract_batch(batch)
            x_num = x_num.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).float().view(-1)
            prediction = self.model(x_num).view(-1)
            loss = self.loss_fn(prediction, y)
            batch_size = len(y)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
        if total_examples == 0:
            raise ValueError("Validation DataLoader is empty")
        return total_loss / total_examples

    def fit(
        self,
        train_loader: DataLoader[Any],
        valid_loader: DataLoader[Any],
        *,
        checkpoint_name: str = "model.pt",
    ) -> FitResult:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / checkpoint_name
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        history: list[dict[str, float]] = []

        try:
            for epoch in range(1, self.config.epochs + 1):
                train_loss = self.train_epoch(train_loader, epoch)
                valid_loss = self.validate_epoch(valid_loader, epoch)
                record = {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "validation_loss": valid_loss,
                }
                history.append(record)
                if self.writer is not None:
                    self.writer.add_scalars(
                        "loss",
                        {"train": train_loss, "validation": valid_loss},
                        epoch,
                    )

                if not math.isfinite(valid_loss):
                    raise FloatingPointError(
                        f"Validation loss became non-finite at epoch {epoch}"
                    )
                improved = valid_loss < best_loss - self.config.min_delta
                if improved:
                    best_loss = valid_loss
                    best_epoch = epoch
                    stale_epochs = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "validation_loss": valid_loss,
                            "trainer_config": asdict(self.config),
                        },
                        checkpoint_path,
                    )
                else:
                    stale_epochs += 1

                LOGGER.info(
                    "epoch=%d train_loss=%.6f validation_loss=%.6f",
                    epoch,
                    train_loss,
                    valid_loss,
                )
                if stale_epochs >= self.config.patience:
                    LOGGER.info("Early stopping at epoch %d", epoch)
                    break
        finally:
            if self.writer is not None:
                self.writer.close()

        if not checkpoint_path.exists():
            raise RuntimeError("Training ended without producing a checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return FitResult(
            train_loss=history[-1]["train_loss"],
            validation_loss=best_loss,
            best_model_path=checkpoint_path.resolve(),
            best_epoch=best_epoch,
            history=tuple(history),
        )

    @torch.inference_mode()
    def predict(self, loader: DataLoader[Any]) -> Tensor:
        self.model.eval()
        predictions = []
        for batch in loader:
            x_num = batch["x_num"].to(self.device, non_blocking=True)
            predictions.append(self.model(x_num).view(-1).cpu())
        return torch.cat(predictions) if predictions else torch.empty(0)
