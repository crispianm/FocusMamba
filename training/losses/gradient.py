"""
Edge-Aware Depth Gradient Smoothness Loss
==========================================

Spatial gradient matching loss (MiDaS / Depth Anything V2 style), adapted
with edge-awareness so the model is free to produce sharp transitions at
genuine depth discontinuities while being penalised for spurious gradients
in smooth regions.

Two modes (controlled by `mode`):
  'matching'  – L1 between spatial gradients of pred and GT (matches paper).
  'smoothness'– Exponentially edge-weighted smoothness penalty on pred only,
                using GT for edge detection (classic ECCV approach).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientSmoothnessLoss(nn.Module):
    """Edge-aware depth gradient loss.

    Args:
        mode:    'matching'   → L1(∇pred, ∇gt)  (recommended, from paper)
                 'smoothness' → exp(-|∇gt|) * |∇pred|  (classic variant)
        scales:  Multi-scale evaluation (averages loss over image pyramid).
        eps:     Numerical guard for log().
        log_space: Compute gradients in log-depth space.
    """

    def __init__(
        self,
        mode: str = "matching",
        scales: int = 4,
        eps: float = 1e-6,
        log_space: bool = True,
    ):
        super().__init__()
        assert mode in ("matching", "smoothness"), f"Unknown mode: {mode}"
        self.mode = mode
        self.scales = scales
        self.eps = eps
        self.log_space = log_space

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_gradients(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute finite-difference spatial gradients.

        Args:
            x: (B, 1, H, W)

        Returns:
            gx: (B, 1, H, W) horizontal gradient (zero-padded right column)
            gy: (B, 1, H, W) vertical gradient   (zero-padded bottom row)
        """
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]   # (B,1,H,W-1)
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]   # (B,1,H-1,W)
        # Pad to restore spatial dims
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return gx, gy

    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, T, H, W) predicted metric depth (positive).
            gt:   (B, 1, T, H, W) ground-truth depth (positive).
            mask: (B, 1, T, H, W) optional validity mask.

        Returns:
            Scalar loss.
        """
        B, C, T, H, W = pred.shape

        # Flatten time into batch → (B*T, 1, H, W)
        p = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        g = gt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        m = (
            mask.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).bool()
            if mask is not None
            else None
        )

        if self.log_space:
            p = torch.log(p.clamp(min=self.eps))
            g = torch.log(g.clamp(min=self.eps))

        total = torch.tensor(0.0, device=pred.device)
        count = 0

        for s in range(self.scales):
            if s > 0:
                p = F.avg_pool2d(p, 2, stride=2)
                g = F.avg_pool2d(g, 2, stride=2)
                if m is not None:
                    m = F.avg_pool2d(m.float(), 2, stride=2) > 0.5

            gx_p, gy_p = self._spatial_gradients(p)
            gx_g, gy_g = self._spatial_gradients(g)

            if self.mode == "matching":
                # L1 between pred gradients and GT gradients
                loss_x = (gx_p - gx_g).abs()
                loss_y = (gy_p - gy_g).abs()
            else:  # smoothness
                # Edge-weighted: suppress pred gradients where GT is smooth
                edge_x = torch.exp(-gx_g.abs())
                edge_y = torch.exp(-gy_g.abs())
                loss_x = edge_x * gx_p.abs()
                loss_y = edge_y * gy_p.abs()

            loss = loss_x + loss_y                    # (B*T,1,H',W')

            if m is not None:
                loss = loss[m]
                if loss.numel() == 0:
                    continue

            total = total + loss.mean()
            count += 1

        return total / max(count, 1)