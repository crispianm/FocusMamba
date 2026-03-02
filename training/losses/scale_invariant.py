"""
Scale-Invariant Logarithmic Depth Loss (SI-Log)
=================================================

The primary depth reconstruction loss for metric depth estimation.
Operates in log-depth space to handle the wide dynamic range of depth values.

SI-log = mean(d²) - λ·mean(d)²
where d = log(pred) - log(gt)

This loss is invariant to global scale shifts while still penalising
local depth errors. Used for both direct supervision and distillation.

Inputs:
    pred: (B, 1, T, H, W) predicted metric depth in metres (positive)
    gt:   (B, 1, T, H, W) ground-truth / teacher metric depth in metres (positive)

Output:
    Scalar loss value.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleInvariantLogLoss(nn.Module):
    """Scale-Invariant Logarithmic depth loss.

    Args:
        lambda_si: Weight for the scale-invariant term (default 0.5).
            0.0 = pure MSE in log space
            1.0 = fully scale-invariant
        eps: Small constant to avoid log(0).
    """

    def __init__(self, lambda_si: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.lambda_si = lambda_si
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute SI-log loss.

        Args:
            pred: (B, 1, T, H, W) predicted depth, must be > 0.
            gt:   (B, 1, T, H, W) target depth, must be > 0.
            mask: (B, 1, T, H, W) optional binary mask (1 = valid pixel).

        Returns:
            Scalar loss.
        """
        # Compute log difference
        log_diff = torch.log(pred.clamp(min=self.eps)) - torch.log(gt.clamp(min=self.eps))

        if mask is not None:
            log_diff = log_diff[mask.bool()]

        if log_diff.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        d_sq_mean = (log_diff ** 2).mean()
        d_mean_sq = (log_diff.mean()) ** 2

        loss = d_sq_mean - self.lambda_si * d_mean_sq
        return loss
