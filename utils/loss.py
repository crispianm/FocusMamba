"""
FocusMamba Loss Functions
==========================

``FocusLoss`` — weighted sum of four components:
    1. L1 Loss               (pixel accuracy)
    2. SSIM Loss              (structural similarity)
    3. Edge-Aware Gradient    (boundary sharpness)
    4. Temporal Gradient Matching (temporal consistency)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude for a (B, 1, H, W) tensor.

    Returns (B, 1, H, W) edge magnitude map.
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return (gx.pow(2) + gy.pow(2)).sqrt()


def _spatial_gradient(x: torch.Tensor) -> torch.Tensor:
    """Spatial gradient magnitude for (B, 1, H, W)."""
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    # Pad to same size
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return (gx.pow(2) + gy.pow(2)).sqrt()


def _ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute 1 - SSIM for (B, 1, H, W) tensors.

    Uses a uniform averaging window (simple box filter) for robustness
    when pytorch_msssim is not available.
    """
    try:
        from pytorch_msssim import ssim
        return 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
    except ImportError:
        pass

    # Fallback: manual SSIM with box filter
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device, dtype=pred.dtype)
    kernel = kernel / kernel.numel()

    mu_p = F.conv2d(pred, kernel, padding=pad)
    mu_t = F.conv2d(target, kernel, padding=pad)
    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t
    sigma_pp = F.conv2d(pred * pred, kernel, padding=pad) - mu_pp
    sigma_tt = F.conv2d(target * target, kernel, padding=pad) - mu_tt
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
        (mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2)
    )
    return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# FocusLoss
# ---------------------------------------------------------------------------

class FocusLoss(nn.Module):
    """Composite focus-map loss.

    Args:
        lambda_l1: Weight for L1 loss.
        lambda_ssim: Weight for SSIM loss.
        lambda_grad: Weight for edge-aware gradient loss.
        lambda_tgm: Weight for temporal gradient matching loss.
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_grad: float = 0.5,
        lambda_tgm: float = 0.2,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_tgm = lambda_tgm

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred: (B, 1, T, H, W) predicted focus maps in [0, 1].
            gt:   (B, 1, T, H, W) ground-truth focus maps in [0, 1].
        Returns:
            Dict with keys: 'total', 'l1', 'ssim', 'grad', 'tgm'.
        """
        B, C, T, H, W = pred.shape

        # ----- Component 1: L1 -----
        loss_l1 = F.l1_loss(pred, gt)

        # ----- Component 2: SSIM (per frame, averaged over T) -----
        ssim_sum = torch.tensor(0.0, device=pred.device)
        for t in range(T):
            ssim_sum = ssim_sum + _ssim_loss(pred[:, :, t], gt[:, :, t])
        loss_ssim = ssim_sum / T

        # ----- Component 3: Edge-Aware Gradient -----
        grad_sum = torch.tensor(0.0, device=pred.device)
        for t in range(T):
            edges_gt = _sobel_edges(gt[:, :, t])          # (B, 1, H, W)
            grad_pred = _spatial_gradient(pred[:, :, t])   # (B, 1, H, W)
            grad_gt = _spatial_gradient(gt[:, :, t])
            grad_sum = grad_sum + (edges_gt * (grad_pred - grad_gt).abs()).mean()
        loss_grad = grad_sum / T

        # ----- Component 4: Temporal Gradient Matching -----
        if T > 1:
            td_pred = pred[:, :, 1:] - pred[:, :, :-1]
            td_gt = gt[:, :, 1:] - gt[:, :, :-1]
            loss_tgm = F.l1_loss(td_pred, td_gt)
        else:
            loss_tgm = torch.tensor(0.0, device=pred.device)

        # ----- Total -----
        total = (
            self.lambda_l1 * loss_l1
            + self.lambda_ssim * loss_ssim
            + self.lambda_grad * loss_grad
            + self.lambda_tgm * loss_tgm
        )

        return {
            "total": total,
            "l1": loss_l1.detach(),
            "ssim": loss_ssim.detach(),
            "grad": loss_grad.detach(),
            "tgm": loss_tgm.detach(),
        }


# ---------------------------------------------------------------------------
# Unit test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running FocusLoss unit test...")
    B, C, T, H, W = 2, 1, 4, 64, 64
    pred = torch.rand(B, C, T, H, W, requires_grad=True)
    gt = torch.rand(B, C, T, H, W)

    criterion = FocusLoss()
    losses = criterion(pred, gt)

    print(f"  total: {losses['total'].item():.4f}")
    print(f"  l1:   {losses['l1'].item():.4f}")
    print(f"  ssim: {losses['ssim'].item():.4f}")
    print(f"  grad: {losses['grad'].item():.4f}")
    print(f"  tgm:  {losses['tgm'].item():.4f}")

    assert losses["total"].item() >= 0, "Loss should be non-negative"

    # Verify back-propagation
    losses["total"].backward()
    assert pred.grad is not None, "Gradient should exist after backward"
    assert pred.grad.abs().sum() > 0, "Gradient should be non-zero"
    print("Unit test passed!")
