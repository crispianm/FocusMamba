"""Lightweight residual depth refiner heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(
            num_groups=max(min(out_channels // 8, 8), 1), num_channels=out_channels
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class RefineNetLite(nn.Module):
    """Refine coarse depth with RGB guidance at a reduced spatial scale."""

    def __init__(
        self,
        *,
        hidden_channels: int = 32,
        refine_scale: float = 0.5,
        predict_edges: bool = False,
    ) -> None:
        super().__init__()
        self.refine_scale = float(refine_scale)
        self.predict_edges = bool(predict_edges)
        in_channels = 4  # RGB + coarse depth
        self.stem = _DepthwiseSeparableBlock(in_channels, hidden_channels)
        self.block = _DepthwiseSeparableBlock(hidden_channels, hidden_channels)
        self.residual_head = nn.Conv2d(
            hidden_channels, 1, kernel_size=3, padding=1, bias=True
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.predict_edges:
            self.edge_head = nn.Conv2d(
                hidden_channels, 1, kernel_size=3, padding=1, bias=True
            )
            nn.init.zeros_(self.edge_head.weight)
            nn.init.zeros_(self.edge_head.bias)
        else:
            self.edge_head = None
        self.output_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        coarse_depth: torch.Tensor,
        rgb_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if coarse_depth.ndim != 5 or rgb_frames.ndim != 5:
            raise ValueError("RefineNetLite expects (B, C, T, H, W) tensors")

        b, _, t, h, w = coarse_depth.shape
        rgb = rgb_frames.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
        depth = coarse_depth.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)

        if self.refine_scale > 0.0 and self.refine_scale != 1.0:
            out_hw = (
                max(1, int(round(h * self.refine_scale))),
                max(1, int(round(w * self.refine_scale))),
            )
            rgb_small = F.interpolate(
                rgb, size=out_hw, mode="bilinear", align_corners=False
            )
            depth_small = F.interpolate(
                depth, size=out_hw, mode="bilinear", align_corners=False
            )
        else:
            rgb_small = rgb
            depth_small = depth

        feats = torch.cat([rgb_small, depth_small], dim=1)
        feats = self.block(self.stem(feats))
        residual = self.residual_head(feats)
        if residual.shape[-2:] != (h, w):
            residual = F.interpolate(
                residual, size=(h, w), mode="bilinear", align_corners=False
            )
        scale = 0.25 * torch.tanh(self.output_scale).to(
            device=residual.device, dtype=residual.dtype
        )
        refined = (
            (depth + scale * residual)
            .reshape(b, t, 1, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        edge_logits = None
        if self.edge_head is not None:
            edge_logits = self.edge_head(feats)
            if edge_logits.shape[-2:] != (h, w):
                edge_logits = F.interpolate(
                    edge_logits, size=(h, w), mode="bilinear", align_corners=False
                )
            edge_logits = (
                edge_logits.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            )

        return refined, edge_logits
