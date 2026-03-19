"""Tiny quality-estimation helpers for degraded video clips."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "QualityNet",
    "QualityTemporalGate",
    "QualityCacheGate",
    "apply_temporal_quality_gate",
]


class QualityNet(nn.Module):
    """Estimate a compact per-frame quality embedding from frames or features."""

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 16,
        hidden_channels: int = 16,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

        blocks: list[nn.Module] = []
        in_ch = int(in_channels)
        hidden = int(hidden_channels)
        depth = max(int(num_blocks), 1)
        for _ in range(depth):
            blocks.append(nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=True))
            blocks.append(nn.GELU())
            in_ch = hidden
        blocks.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*blocks)
        self.proj = nn.Linear(hidden, self.embedding_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        frames: torch.Tensor,
        *,
        batch_size: int | None = None,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        if frames.ndim == 5:
            b, c, t, h, w = frames.shape
            x = frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        elif frames.ndim == 4:
            if batch_size is None or num_frames is None:
                raise ValueError(
                    "QualityNet expects batch_size and num_frames when given flattened (B*T, C, H, W) input."
                )
            bt, c, h, w = frames.shape
            b = int(batch_size)
            t = int(num_frames)
            if bt != b * t:
                raise ValueError(
                    f"Flattened QualityNet input has {bt} frames but batch_size*num_frames={b*t}."
                )
            x = frames
        else:
            raise ValueError(
                f"QualityNet expects (B, C, T, H, W) or (B*T, C, H, W), got {tuple(frames.shape)}"
            )

        x = self.encoder(x).flatten(1)
        x = self.proj(x)
        return x.reshape(b, t, self.embedding_dim)


class QualityTemporalGate(nn.Module):
    """Map quality embeddings to multiplicative temporal gates."""

    def __init__(self, embedding_dim: int = 16, out_channels: int = 1) -> None:
        super().__init__()
        self.proj = nn.Linear(int(embedding_dim), int(out_channels))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.proj(embedding)


class QualityCacheGate(nn.Module):
    """Map quality embeddings to multiplicative cache-gating weights."""

    def __init__(self, channels: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(int(embedding_dim), int(channels))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        current: torch.Tensor,
        cached: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        del current, cached
        if embedding.ndim == 3:
            summary = embedding.mean(dim=1)
        elif embedding.ndim == 2:
            summary = embedding
        else:
            raise ValueError(
                f"QualityCacheGate expects (B, T, D) or (B, D) embedding, got {tuple(embedding.shape)}"
            )
        return 1.0 + self.proj(summary)


def apply_temporal_quality_gate(sequence: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 5:
        raise ValueError(
            f"apply_temporal_quality_gate expects (B, C, T, H, W), got {tuple(sequence.shape)}"
        )
    if gate.ndim != 3:
        raise ValueError(
            f"apply_temporal_quality_gate expects gate with shape (B, T, Cg), got {tuple(gate.shape)}"
        )
    return sequence * gate.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
