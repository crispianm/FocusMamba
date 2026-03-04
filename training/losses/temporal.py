"""
Temporal Gradient Matching (TGM) Loss
=======================================

From: "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos"
      Chen et al., 2025 (arXiv:2501.12375)

Core formula (Eq. 3 in paper):

    L_TGM = 1/(N-1) * Σ_i  ||  |d_{i+1} - d_i|  -  |g_{i+1} - g_i|  ||_1

where d_i, g_i are predicted and GT depth at frame i.

Key insight: the loss only fires where GT temporal change is small
(|g_{i+1} - g_i| < threshold). This avoids penalising legitimate
depth discontinuities at moving-object edges and scene boundaries.

Metric-depth adaptation
------------------------
The VDA paper computes TGM on scale-shift aligned depth values with a
threshold of 0.05. This implementation defaults to that behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConsistencyLoss(nn.Module):
    """Temporal Gradient Matching loss (TGM, Eq. 3 of Video Depth Anything).

    Args:
        threshold:  GT change threshold for the stability mask.
                    Paper uses 0.05.
        log_space:  Optional log-depth mode. Defaults to False for VDA parity.
        eps:        Numerical guard for log().
    """

    def __init__(
        self,
        threshold: float = 0.05,
        log_space: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.threshold = threshold
        self.log_space = log_space
        self.eps = eps

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
            mask: (B, 1, T, H, W) optional binary validity mask (1 = valid).

        Returns:
            Scalar TGM loss.
        """
        T = pred.shape[2]
        if T <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if self.log_space:
            p = torch.log(pred.clamp(min=self.eps))
            g = torch.log(gt.clamp(min=self.eps))
        else:
            p = pred
            g = gt

        # Temporal finite differences  →  (B, 1, T-1, H, W)
        dp = p[:, :, 1:] - p[:, :, :-1]
        dg = g[:, :, 1:] - g[:, :, :-1]

        # ── Stability mask ──────────────────────────────────────────────────
        # Only regions where GT is temporally smooth (static background).
        # Excludes object edges, dynamic objects, occlusion boundaries.
        stable: torch.Tensor = dg.abs() < self.threshold   # (B,1,T-1,H,W)

        # Both adjacent frames must be valid
        if mask is not None:
            mb = mask.bool()
            valid_pair = mb[:, :, 1:] & mb[:, :, :-1]
            stable = stable & valid_pair

        if not stable.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # L1 between |pred temporal change| and |GT temporal change|
        loss_map = (dp.abs() - dg.abs()).abs()              # (B,1,T-1,H,W)
        return loss_map[stable].mean()
