from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ResNetConfig:
    d_in: int = 158
    depth: int = 4
    width: int = 256
    hidden_factor: float = 2.0
    dropout_first: float = 0.25
    dropout_second: float = 0.0
    d_out: int = 1

    @property
    def hidden_width(self) -> int:
        return int(self.width * self.hidden_factor)

    def __post_init__(self) -> None:
        if min(self.d_in, self.depth, self.width, self.d_out) <= 0:
            raise ValueError("dimensions and depth must be positive")
        if self.hidden_factor <= 0:
            raise ValueError("hidden_factor must be positive")
        for value in (self.dropout_first, self.dropout_second):
            if not 0.0 <= value < 1.0:
                raise ValueError("dropout values must be in [0, 1)")


class ResNetBlock(nn.Module):
    """RTDL-style pre-normalization residual block.

    The first projection expands the representation to ``hidden_width``.
    The second projection returns to ``width`` before the residual addition.
    RTDL applies dropout after the activation and optionally after the second
    linear layer; keeping these rates separate matches the official design.
    """

    def __init__(
        self,
        width: int,
        hidden_width: int,
        dropout_first: float,
        dropout_second: float,
    ) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(width)
        self.linear_first = nn.Linear(width, hidden_width)
        self.activation = nn.ReLU()
        self.dropout_first = nn.Dropout(dropout_first)
        self.linear_second = nn.Linear(hidden_width, width)
        self.dropout_second = nn.Dropout(dropout_second)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.normalization(x)
        x = self.linear_first(x)
        x = self.activation(x)
        x = self.dropout_first(x)
        x = self.linear_second(x)
        x = self.dropout_second(x)
        return residual + x


class ResNet(nn.Module):
    """Residual MLP for tabular regression, following the RTDL architecture."""

    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.d_in, config.width)
        self.blocks = nn.Sequential(
            *[
                ResNetBlock(
                    config.width,
                    config.hidden_width,
                    config.dropout_first,
                    config.dropout_second,
                )
                for _ in range(config.depth)
            ]
        )
        # The official prediction head normalizes and activates once more.
        self.head = nn.Sequential(
            nn.BatchNorm1d(config.width),
            nn.ReLU(),
            nn.Linear(config.width, config.d_out),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_num: Tensor) -> Tensor:
        output = self.head(self.blocks(self.input(x_num)))
        return output.squeeze(-1) if self.config.d_out == 1 else output
