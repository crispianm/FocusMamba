"""
Edge-Aware Depth Gradient Smoothness Loss
===========================================

Encourages smooth depth predictions while preserving sharp edges.

TO BE IMPLEMENTED by the user. This stub defines the expected interface.

Expected interface:
    loss_fn = GradientSmoothnessLoss()
    loss = loss_fn(pred_depth, gt_depth)

Inputs:
    pred: (B, 1, T, H, W) predicted metric depth
    gt:   (B, 1, T, H, W) ground-truth depth (used for edge detection)

Output:
    Scalar loss value.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientSmoothnessLoss(nn.Module):
    """Edge-aware depth gradient smoothness loss.

    Penalises depth gradients in smooth regions while allowing sharp
    transitions at image edges (detected via Sobel filter on GT depth).

    TODO: Implement. Suggested approach:
        1. Compute spatial gradients of predicted depth (∂d/∂x, ∂d/∂y)
        2. Compute edge weights from GT depth or RGB image (Sobel)
        3. Weight the gradient penalty inversely by edge strength
        4. loss = mean(exp(-|edge|) * |∇depth_pred|)
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
        raise NotImplementedError("Implement edge-aware gradient smoothness loss")
