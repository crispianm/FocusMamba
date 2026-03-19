"""Training-only auxiliary reconstruction head."""

from __future__ import annotations

import torch
import torch.nn as nn


class AuxRestorationDecoder(nn.Module):
    """Predict clean RGB from degraded RGB plus coarse depth."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        in_channels = 4  # RGB + depth
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=max(hidden_channels // 4, 1),
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.output_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self, degraded_frames: torch.Tensor, coarse_depth: torch.Tensor
    ) -> torch.Tensor:
        if degraded_frames.ndim != 5 or coarse_depth.ndim != 5:
            raise ValueError("AuxRestorationDecoder expects (B, C, T, H, W) tensors")
        b, _, t, h, w = degraded_frames.shape
        rgb = degraded_frames.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
        depth = coarse_depth.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        recon_residual = self.net(torch.cat([rgb, depth], dim=1))
        scale = 0.25 * torch.tanh(self.output_scale).to(
            device=rgb.device, dtype=rgb.dtype
        )
        recon = (rgb + scale * recon_residual).clamp(0.0, 1.0)
        return recon.reshape(b, t, 3, h, w).permute(0, 2, 1, 3, 4).contiguous()
