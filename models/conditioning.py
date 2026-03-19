"""Conditioning helpers for degradation-aware FocusMamba variants."""

from __future__ import annotations

import torch
import torch.nn as nn


class DegradationConditioningMLP(nn.Module):
    """Map a compact degradation descriptor to scan-weight deltas."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 32, output_dim: int = 4
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        hidden_dim = max(int(hidden_dim), self.output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected conditioning tensor with shape (B, K), got {tuple(x.shape)}"
            )
        return 0.25 * torch.tanh(self.net(x))
