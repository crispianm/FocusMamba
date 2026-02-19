"""
ROI Conditioning for FocusMamba
================================

Generates a soft Gaussian ROI mask from a normalised bounding box and
projects it to an additive spatial bias that is added to patch embeddings
*before* the first Mamba block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIConditioner(nn.Module):
    """Purely spatial additive ROI conditioning.

    1. Generate a soft 2D Gaussian mask from the ROI box (B,4).
    2. Project through a small CNN to (B, embed_dim, H_tok, W_tok).
    3. Return the additive bias to be summed with patch embeddings.

    Args:
        embed_dim: Dimension of patch embeddings.
        patch_size: Spatial patch size used by the tubelet embedding,
            needed to down-sample the mask to token-grid resolution.
    """

    def __init__(self, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, embed_dim, kernel_size=3, padding=1),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _make_gaussian_mask(
        roi: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Create soft Gaussian masks of shape (B, 1, H, W).

        Args:
            roi: (B, 4) normalised [x1, y1, x2, y2] in [0, 1].
            H, W: Spatial dimensions (token-grid resolution).
        """
        B = roi.shape[0]
        device = roi.device

        # Centre and size in pixel (token-grid) coords
        x1 = roi[:, 0] * W
        y1 = roi[:, 1] * H
        x2 = roi[:, 2] * W
        y2 = roi[:, 3] * H

        cx = (x1 + x2) / 2.0  # (B,)
        cy = (y1 + y2) / 2.0
        sx = (x2 - x1).clamp(min=1.0) / 2.0  # sigma proportional to box size
        sy = (y2 - y1).clamp(min=1.0) / 2.0

        # Coordinate grids
        yy = torch.arange(H, device=device, dtype=torch.float32)
        xx = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")  # (H, W)

        # Broadcast: (B, 1, 1) vs (H, W)
        cx = cx.view(B, 1, 1)
        cy = cy.view(B, 1, 1)
        sx = sx.view(B, 1, 1)
        sy = sy.view(B, 1, 1)

        gauss = torch.exp(
            -0.5 * (((grid_x.unsqueeze(0) - cx) / sx) ** 2 +
                     ((grid_y.unsqueeze(0) - cy) / sy) ** 2)
        )  # (B, H, W)
        return gauss.unsqueeze(1)  # (B, 1, H, W)

    # ------------------------------------------------------------------
    def forward(
        self, roi: torch.Tensor, H_tok: int, W_tok: int
    ) -> torch.Tensor:
        """
        Args:
            roi: (B, 4) normalised ROI box.
            H_tok, W_tok: Token-grid spatial dimensions
                (= image_H // patch_size, image_W // patch_size).
        Returns:
            (B, embed_dim, H_tok, W_tok) additive bias.
        """
        mask = self._make_gaussian_mask(roi, H_tok, W_tok)  # (B, 1, H_tok, W_tok)
        return self.projection(mask)  # (B, embed_dim, H_tok, W_tok)
