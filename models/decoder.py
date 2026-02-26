"""
"""FocusMamba Decoder — Metric Depth
======================================

A 4-stage UNet decoder that mirrors the encoder's multi-scale hierarchy.
Outputs metric depth (via exp(log_depth) for positivity enforcement) and
optionally per-pixel aleatoric uncertainty.

Each decoder stage:
    1. Upsample by 2x (trilinear + 1×1×1 conv to halve channels).
    2. Concatenate skip connection from encoder (→ 2C channels).
    3. Project 2C → C with 1×1×1 conv.
    4. Apply CrossScanSpatialMambaBlock (operates at C channels).
    5. Apply 3-D depthwise separable conv for local smoothing.

Channel dims per stage:
    Stage 3 (from bottleneck): 768 → concat(768+384)=1152 → project → 384
    Stage 2:                   384 → concat(384+192)=576   → project → 192
    Stage 1:                   192 → concat(192+96)=288    → project → 96
    Stage 0:                   96  → (no skip at input res) → 96

Final head predicts log-depth (exponentiated for metric output) and
optionally per-pixel uncertainty via separate head.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import SpatialMambaBlock


# ---------------------------------------------------------------------------
# 3-D Depthwise Separable Conv
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv3d(nn.Module):
    """3-D depthwise separable convolution for local spatiotemporal smoothing."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv3d(
            channels, channels, kernel_size,
            padding=pad, groups=channels, bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, 1, bias=True)
        self.norm = nn.GroupNorm(1, channels)  # instance-norm-like
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W)"""
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


# ---------------------------------------------------------------------------
# Single Decoder Stage
# ---------------------------------------------------------------------------

class DecoderStage(nn.Module):
    """One decoder stage: upsample → concat skip → project → Mamba → DWConv."""

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.has_skip = skip_dim > 0

        # Channel reduction after upsample (in_dim → out_dim)
        self.upsample_conv = nn.Conv3d(in_dim, out_dim, kernel_size=1)

        if self.has_skip:
            # Project concatenated channels: out_dim + skip_dim → out_dim
            self.proj = nn.Conv3d(out_dim + skip_dim, out_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.spatial_mamba = SpatialMambaBlock(out_dim, d_state, d_conv, expand)
        self.dw_conv = DepthwiseSeparableConv3d(out_dim, kernel_size=3)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C_in) from previous decoder stage / bottleneck.
            skip: (B, T, H_s, W_s, C_skip) encoder skip (or None for stage 0).
        Returns:
            (B, T, H*2, W*2, C_out)
        """
        B, T, H, W, C = x.shape

        # 1. Upsample spatially by 2x
        x5d = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x5d = F.interpolate(
            x5d, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False,
        )
        x5d = self.upsample_conv(x5d)  # (B, out_dim, T, H*2, W*2)

        # 2. Concat skip
        if self.has_skip and skip is not None:
            skip5d = skip.permute(0, 4, 1, 2, 3)  # (B, C_skip, T, H_s, W_s)
            # Ensure spatial dims match
            if skip5d.shape[3:] != x5d.shape[3:]:
                skip5d = F.interpolate(
                    skip5d, size=x5d.shape[2:], mode="trilinear", align_corners=False,
                )
            x5d = torch.cat([x5d, skip5d], dim=1)  # (B, out_dim+skip_dim, ...)
            x5d = self.proj(x5d)  # → (B, out_dim, T, H', W')

        # Convert to (B, T, H', W', C)
        x_tokens = x5d.permute(0, 2, 3, 4, 1)

        # 4. Spatial Mamba
        x_tokens = self.spatial_mamba(x_tokens)

        # 5. Depthwise separable conv (operates in (B,C,T,H,W))
        x5d = x_tokens.permute(0, 4, 1, 2, 3)
        x5d = self.dw_conv(x5d)
        return x5d.permute(0, 2, 3, 4, 1)


# ---------------------------------------------------------------------------
# Full Decoder with Temporal Smoothing Head
# ---------------------------------------------------------------------------

class FocusMambaDecoder(nn.Module):
    """UNet decoder for FocusMamba — metric depth output.

    Predicts log-depth which is exponentiated to enforce positive metric depth.
    Optionally predicts per-pixel aleatoric uncertainty.

    Args:
        embed_dim: Base embedding dimension (same as encoder stage-0 dim).
        d_state, d_conv, expand: Mamba SSM hyper-params.
        predict_uncertainty: If True, adds a separate uncertainty head.
    """

    def __init__(
        self,
        embed_dim: int = 96,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        predict_uncertainty: bool = False,
    ):
        super().__init__()
        self.predict_uncertainty = predict_uncertainty

        # Channel dims: stage-0=96, stage-1=192, stage-2=384, bottleneck=768
        dims = [embed_dim * (2 ** i) for i in range(4)]  # [96, 192, 384, 768]

        # Decoder stages (high to low index = coarse to fine)
        # Stage 3: bottleneck(768) → upsample → concat skip2(384) → 384
        # Stage 2: 384 → upsample → concat skip1(192) → 192
        # Stage 1: 192 → upsample → concat skip0(96) → 96
        # Stage 0: 96 → no skip → 96  (final res)
        self.stages = nn.ModuleList([
            DecoderStage(dims[3], dims[2], dims[2], d_state, d_conv, expand),  # 768→384
            DecoderStage(dims[2], dims[1], dims[1], d_state, d_conv, expand),  # 384→192
            DecoderStage(dims[1], dims[0], dims[0], d_state, d_conv, expand),  # 192→96
            DecoderStage(dims[0], 0,       dims[0], d_state, d_conv, expand),  # 96→96
        ])

        # Temporal smoothing conv (1-D along T, kernel=3)
        self.temporal_smooth = nn.Conv3d(
            dims[0], dims[0],
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=dims[0],
        )

        # Metric depth head: predicts log-depth, exponentiated for positivity
        self.depth_head = nn.Conv3d(dims[0], 1, kernel_size=1)

        # Optional uncertainty head: predicts log-variance
        if predict_uncertainty:
            self.uncertainty_head = nn.Conv3d(dims[0], 1, kernel_size=1)

    def forward(
        self,
        skips: List[torch.Tensor],
        bottleneck: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            skips: list of encoder skips [stage0, stage1, stage2], high-res first.
            bottleneck: (B, T, H_b, W_b, C_b).
        Returns:
            dict with:
                'depth': (B, 1, T, H_out, W_out) metric depth in metres (positive).
                'uncertainty': (B, 1, T, H_out, W_out) log-variance (if enabled).
        """
        x = bottleneck  # (B, T, H, W, 768)

        # skips list: [stage0_skip, stage1_skip, stage2_skip]
        # We consume them in reverse order (coarse → fine)
        reversed_skips = list(reversed(skips))  # [stage2, stage1, stage0]

        for i, stage in enumerate(self.stages):
            if i < len(reversed_skips):
                x = stage(x, reversed_skips[i])
            else:
                x = stage(x, None)

        # x: (B, T, H, W, 96) → (B, 96, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)

        # Temporal smoothing
        x = self.temporal_smooth(x)

        # Metric depth head: predict log-depth, exponentiate for positivity
        log_depth = self.depth_head(x)  # (B, 1, T, H, W)
        depth = torch.exp(log_depth)    # always positive, in metres

        outputs = {"depth": depth}

        # Optional uncertainty head
        if self.predict_uncertainty:
            log_variance = self.uncertainty_head(x)  # (B, 1, T, H, W)
            outputs["uncertainty"] = log_variance

        return outputs
