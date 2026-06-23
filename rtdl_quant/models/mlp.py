from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MLPConfig:
    d_in: int = 158
    hidden_dim: int = 256
    hidden_dims: tuple[int, ...] | None = None
    dropout: float = 0.1
    d_out: int = 1

    @property
    def layer_widths(self) -> tuple[int, int]:
        if self.hidden_dims is None:
            return (self.hidden_dim, self.hidden_dim)
        return self.hidden_dims

    def __post_init__(self) -> None:
        if self.d_in <= 0 or self.d_out <= 0:
            raise ValueError("d_in and d_out must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if len(self.layer_widths) != 2 or any(d <= 0 for d in self.layer_widths):
            raise ValueError("hidden_dims must contain exactly two positive widths")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class MLP(nn.Module):
    """Two-hidden-layer MLP baseline for Alpha158 regression."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        h1, h2 = config.layer_widths
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.d_in, h1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h2, config.d_out),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_num: Tensor) -> Tensor:
        output = self.network(x_num)
        return output.squeeze(-1) if self.config.d_out == 1 else output
