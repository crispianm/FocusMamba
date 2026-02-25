"""
FocusMamba Encoder
===================

Components:
- ``TubeletEmbedding``: 3-D Conv → sinusoidal positional encoding.
- ``EncoderStage``: Stack of alternating SpatialMamba + TemporalMamba blocks.
- ``FocusMambaEncoder``: Multi-scale encoder with skip connections.

Default medium-model configuration:
    Stage 0 (res/4):   embed=96,  depth=2
    Stage 1 (res/8):   embed=192, depth=2
    Stage 2 (res/16):  embed=384, depth=4
    Stage 3 (res/32):  embed=768, depth=2  (bottleneck)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .mamba_block import SpatialMambaBlock, TemporalMambaBlock


# ---------------------------------------------------------------------------
# Sinusoidal 3-D positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_pos_encoding_3d(
    T: int, H: int, W: int, C: int, device: torch.device
) -> torch.Tensor:
    """Create a (1, T, H, W, C) sinusoidal positional encoding.

    The channel dimension is split into three equal parts for temporal,
    vertical, and horizontal positions.  Remaining channels (if C is not
    divisible by 3) are padded with zeros.
    """
    assert C >= 6, "Embedding dim must be >= 6 for 3-D sinusoidal PE"
    c_per_axis = C // 3

    def _encode_axis(length: int, d: int) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=torch.float32) * -(math.log(10000.0) / d)
        )
        pe = torch.zeros(length, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d // 2] if d % 2 else div)
        return pe

    pe_t = _encode_axis(T, c_per_axis)  # (T, c)
    pe_h = _encode_axis(H, c_per_axis)
    pe_w = _encode_axis(W, C - 2 * c_per_axis)  # absorb remainder

    # Broadcast to (T, H, W, C)
    pe = torch.zeros(T, H, W, C, device=device)
    pe[:, :, :, :c_per_axis] += pe_t[:, None, None, :]
    pe[:, :, :, c_per_axis : 2 * c_per_axis] += pe_h[None, :, None, :]
    pe[:, :, :, 2 * c_per_axis :] += pe_w[None, None, :, :]
    return pe.unsqueeze(0)  # (1, T, H, W, C)


# ---------------------------------------------------------------------------
# Tubelet Embedding
# ---------------------------------------------------------------------------

class TubeletEmbedding(nn.Module):
    """3-D convolutional tubelet embedding with positional encoding.

    Maps (B, C_in, T, H, W) → (B, T', H', W', embed_dim) where
    T' = T // t_patch, H' = H // patch_size, W' = W // patch_size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        patch_size: int = 4,
        t_patch: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.t_patch = t_patch
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(t_patch, patch_size, patch_size),
            stride=(t_patch, patch_size, patch_size),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) input video frames.
        Returns:
            (B, T', H', W', embed_dim)
        """
        B, C, T, H, W = x.shape
        tokens = self.proj(x)  # (B, embed_dim, T', H', W')
        _, _, Tp, Hp, Wp = tokens.shape
        tokens = tokens.permute(0, 2, 3, 4, 1)  # (B, T', H', W', embed_dim)

        # Sinusoidal 3-D positional encoding
        pe = sinusoidal_pos_encoding_3d(Tp, Hp, Wp, self.embed_dim, x.device)
        tokens = tokens + pe

        return tokens


# ---------------------------------------------------------------------------
# Encoder Stage
# ---------------------------------------------------------------------------

class EncoderStage(nn.Module):
    """A single encoder stage: ``depth`` pairs of Spatial + Temporal Mamba blocks.

    Args:
        dim: Channel dimension for this stage.
        depth: Number of (Spatial, Temporal) block pairs.
        d_state, d_conv, expand: Mamba hyper-parameters.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(SpatialMambaBlock(dim, d_state, d_conv, expand))
            layers.append(TemporalMambaBlock(dim, d_state, d_conv, expand))
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C) → (B, T, H, W, C)"""
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Spatial Downsample (strided 3-D Conv, spatial only)
# ---------------------------------------------------------------------------

class SpatialDownsample(nn.Module):
    """Downsample spatial dims by 2x, double channels. Temporal dim untouched."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv3d(
            in_dim, out_dim,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C) → (B, T, H/2, W/2, out_dim)"""
        x = x.permute(0, 4, 1, 2, 3)   # (B, C, T, H, W)
        x = self.conv(x)
        return x.permute(0, 2, 3, 4, 1)  # (B, T, H', W', C')


# ---------------------------------------------------------------------------
# Full Multi-Scale Encoder
# ---------------------------------------------------------------------------

class FocusMambaEncoder(nn.Module):
    """Multi-scale encoder with skip connections.

    Returns a list of skip features (one per stage, high-res first)
    and the bottleneck tensor.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] | None = None,
        patch_size: int = 4,
        t_patch: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 4, 2]
        self.num_stages = len(depths)

        # Tubelet embedding
        self.embed = TubeletEmbedding(in_channels, embed_dim, patch_size, t_patch)

        # Build stages and downsamples
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dim = embed_dim
        for i, depth in enumerate(depths):
            self.stages.append(EncoderStage(dim, depth, d_state, d_conv, expand))
            if i < len(depths) - 1:
                next_dim = dim * 2
                self.downsamples.append(SpatialDownsample(dim, next_dim))
                dim = next_dim

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, C, T, H, W) input video.
        Returns:
            skips: list of (B, T', H_i, W_i, C_i), high-res first.
            bottleneck: (B, T', H_last, W_last, C_last).
        """
        tokens = self.embed(x)  # (B, T', H', W', embed_dim)

        skips: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            tokens = stage(tokens)
            if i < self.num_stages - 1:
                skips.append(tokens)  # store skip before downsample
                tokens = self.downsamples[i](tokens)

        bottleneck = tokens
        return skips, bottleneck
