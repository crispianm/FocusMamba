"""
FocusTransformer — Bidirectional Transformer Baseline
======================================================

Mirrors the FocusMamba architecture exactly:
    TubeletEmbedding → Encoder → Decoder → Head

The *only* difference is that Mamba SSM blocks are replaced with standard
bidirectional Multi-Head Self-Attention + MLP blocks.  The feature extractor
(tubelet embedding, spatial downsamples/upsamples) and prediction head
(temporal smoothing conv + sigmoid) are identical.

This enables a fair 1:1 comparison where any performance difference is
attributable to the sequence modelling core (causal Mamba vs bidirectional
Transformer).

Inputs:
    frames: (B, C, T, H, W) — video clip (RGB, float [0,1])

Outputs:
    focus_map: (B, 1, T, H, W) — soft focus map in [0, 1]
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TubeletEmbedding, SpatialDownsample
from .decoder import DepthwiseSeparableConv3d
from .transformer_block import (
    SpatialTransformerBlock,
    TemporalTransformerBlock,
)


# ---------------------------------------------------------------------------
# Transformer Encoder Stage
# ---------------------------------------------------------------------------

class TransformerEncoderStage(nn.Module):
    """A single encoder stage: ``depth`` pairs of Spatial + Temporal Transformer blocks.

    Drop-in replacement for ``EncoderStage`` from encoder.py.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        # Accept Mamba-style kwargs for API compatibility
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(SpatialTransformerBlock(dim, num_heads, mlp_ratio))
            layers.append(TemporalTransformerBlock(dim, num_heads, mlp_ratio))
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C) → (B, T, H, W, C)"""
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

class FocusTransformerEncoder(nn.Module):
    """Multi-scale Transformer encoder with skip connections.

    Same hierarchy as FocusMambaEncoder but using Transformer blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] | None = None,
        patch_size: int = 4,
        t_patch: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        # Keep d_state/d_conv/expand in signature for config compatibility
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 4, 2]
        self.num_stages = len(depths)

        # Shared tubelet embedding (identical to Mamba variant)
        self.embed = TubeletEmbedding(in_channels, embed_dim, patch_size, t_patch)

        # Build stages and downsamples
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dim = embed_dim
        for i, depth in enumerate(depths):
            # Scale heads proportionally to dim so head_dim stays ~32-48
            stage_heads = max(1, dim // 32)
            self.stages.append(
                TransformerEncoderStage(dim, depth, num_heads=stage_heads, mlp_ratio=mlp_ratio)
            )
            if i < len(depths) - 1:
                next_dim = dim * 2
                self.downsamples.append(SpatialDownsample(dim, next_dim))
                dim = next_dim

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        tokens = self.embed(x)
        skips: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            tokens = stage(tokens)
            if i < self.num_stages - 1:
                skips.append(tokens)
                tokens = self.downsamples[i](tokens)
        return skips, tokens


# ---------------------------------------------------------------------------
# Transformer Decoder Stage
# ---------------------------------------------------------------------------

class TransformerDecoderStage(nn.Module):
    """One decoder stage: upsample → concat skip → project → Transformer → DWConv.

    Drop-in replacement for ``DecoderStage`` from decoder.py.
    """

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.has_skip = skip_dim > 0

        self.upsample_conv = nn.Conv3d(in_dim, out_dim, kernel_size=1)

        if self.has_skip:
            self.proj = nn.Conv3d(out_dim + skip_dim, out_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

        stage_heads = max(1, out_dim // 32)
        self.spatial_transformer = SpatialTransformerBlock(
            out_dim, num_heads=stage_heads, mlp_ratio=mlp_ratio,
        )
        self.dw_conv = DepthwiseSeparableConv3d(out_dim, kernel_size=3)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape

        # 1. Upsample spatially by 2x
        x5d = x.permute(0, 4, 1, 2, 3)
        x5d = F.interpolate(x5d, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        x5d = self.upsample_conv(x5d)

        # 2. Concat skip
        if self.has_skip and skip is not None:
            skip5d = skip.permute(0, 4, 1, 2, 3)
            if skip5d.shape[3:] != x5d.shape[3:]:
                skip5d = F.interpolate(skip5d, size=x5d.shape[2:], mode="trilinear", align_corners=False)
            x5d = torch.cat([x5d, skip5d], dim=1)
            x5d = self.proj(x5d)

        x_tokens = x5d.permute(0, 2, 3, 4, 1)

        # 4. Spatial Transformer
        x_tokens = self.spatial_transformer(x_tokens)

        # 5. Depthwise separable conv
        x5d = x_tokens.permute(0, 4, 1, 2, 3)
        x5d = self.dw_conv(x5d)
        return x5d.permute(0, 2, 3, 4, 1)


# ---------------------------------------------------------------------------
# Transformer Decoder
# ---------------------------------------------------------------------------

class FocusTransformerDecoder(nn.Module):
    """UNet decoder using Transformer blocks.  Mirrors FocusMambaDecoder exactly."""

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        dims = [embed_dim * (2 ** i) for i in range(4)]

        self.stages = nn.ModuleList([
            TransformerDecoderStage(dims[3], dims[2], dims[2], num_heads=max(1, dims[2]//32), mlp_ratio=mlp_ratio),
            TransformerDecoderStage(dims[2], dims[1], dims[1], num_heads=max(1, dims[1]//32), mlp_ratio=mlp_ratio),
            TransformerDecoderStage(dims[1], dims[0], dims[0], num_heads=max(1, dims[0]//32), mlp_ratio=mlp_ratio),
            TransformerDecoderStage(dims[0], 0,       dims[0], num_heads=max(1, dims[0]//32), mlp_ratio=mlp_ratio),
        ])

        self.temporal_smooth = nn.Conv3d(
            dims[0], dims[0],
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=dims[0],
        )
        self.head = nn.Conv3d(dims[0], 1, kernel_size=1)

    def forward(
        self,
        skips: List[torch.Tensor],
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        x = bottleneck
        reversed_skips = list(reversed(skips))

        for i, stage in enumerate(self.stages):
            if i < len(reversed_skips):
                x = stage(x, reversed_skips[i])
            else:
                x = stage(x, None)

        x = x.permute(0, 4, 1, 2, 3)
        x = self.temporal_smooth(x)
        out = torch.sigmoid(self.head(x))
        return out


# ---------------------------------------------------------------------------
# Full FocusTransformer Model
# ---------------------------------------------------------------------------

class FocusTransformer(nn.Module):
    """End-to-end bidirectional Transformer model for focus map prediction.

    Architecture mirrors FocusMamba exactly — same tubelet embedding, same
    multi-scale UNet hierarchy, same temporal smoothing head.  Only the
    sequence modelling core differs (MHSA vs Mamba SSM).

    Args:
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Base embedding dimension.
        depths: Number of Spatial+Temporal block pairs per encoder stage.
        patch_size: Spatial patch size for tubelet embedding.
        t_patch: Temporal patch size for tubelet embedding.
        num_heads: Base number of attention heads (auto-scaled per stage).
        mlp_ratio: MLP hidden dimension multiplier.
        d_state: Unused (kept for config compatibility).
        d_conv: Unused (kept for config compatibility).
        expand: Unused (kept for config compatibility).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] | None = None,
        patch_size: int = 4,
        t_patch: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 4, 2]

        self.encoder = FocusTransformerEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            patch_size=patch_size,
            t_patch=t_patch,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.decoder = FocusTransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.patch_size = patch_size
        self.t_patch = t_patch

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, C, T, H, W) float32 in [0,1].
        Returns:
            focus_map: (B, 1, T, H, W) in [0,1].
        """
        B, C, T, H, W = frames.shape
        skips, bottleneck = self.encoder(frames)
        focus_map = self.decoder(skips, bottleneck)

        if focus_map.shape[2:] != (T, H, W):
            focus_map = F.interpolate(
                focus_map, size=(T, H, W), mode="trilinear", align_corners=False,
            )
        return focus_map

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(
        self, input_shape: Tuple[int, ...] = (1, 3, 8, 256, 256)
    ) -> int:
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy = torch.zeros(input_shape, device=next(self.parameters()).device)
            return int(FlopCountAnalysis(self, (dummy,)).total())
        except Exception:
            B, C, T, H, W = input_shape
            n_tokens = (T // self.t_patch) * (H // self.patch_size) * (W // self.patch_size)
            return 2 * self.count_parameters() * n_tokens


if __name__ == "__main__":
    model = FocusTransformer(
        in_channels=3, embed_dim=96, depths=[2, 2, 4, 2],
        patch_size=4, t_patch=2,
    )
    print(f"Parameters: {model.count_parameters():,}")

    x = torch.randn(1, 3, 8, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (1, 1, 8, 256, 256), f"Shape mismatch: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output out of [0,1] range"
    print("Smoke test passed!")
