"""
Temporal Consistency Loss
==========================

Encourages temporally smooth depth predictions across frames.

TO BE IMPLEMENTED by the user. This stub defines the expected interface.

Expected interface:
    loss_fn = TemporalConsistencyLoss()
    loss = loss_fn(pred_depth, gt_depth)

Inputs:
    pred: (B, 1, T, H, W) predicted metric depth
    gt:   (B, 1, T, H, W) ground-truth / teacher depth

Output:
    Scalar loss value.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video depth.

    Penalises temporal flickering in depth predictions that doesn't
    correspond to actual scene motion.

    TODO: Implement. Suggested approaches:
        1. L1 on temporal gradients: |∂pred/∂t - ∂gt/∂t|
        2. Optical-flow-warped consistency (requires flow estimator)
        3. OPW (Ordered Point-Wise) temporal error
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, T, H, W) predicted depth.
            gt:   (B, 1, T, H, W) ground-truth depth.
            mask: (B, 1, T, H, W) optional validity mask.
        Returns:
            Scalar loss.
        """
        raise NotImplementedError("Implement temporal consistency loss")
