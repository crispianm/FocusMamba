"""
FocusMamba Evaluation Metrics
==============================

All metrics operate on (B, 1, T, H, W) prediction / ground-truth tensors.

Metrics:
    - MAE:  Mean Absolute Error.
    - Weighted F-measure (Fβω, β=0.3): threshold-free continuous F-measure.
    - TSS:  Temporal Stability Score.
    - SSIM: Structural Similarity (per-frame, averaged over T).
    - PSNR: Peak Signal-to-Noise Ratio.
    - LPIPS: Learned Perceptual Image Patch Similarity (requires `lpips` pkg).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def mae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    return (pred - gt).abs().mean()


def psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (higher is better)."""
    mse = F.mse_loss(pred, gt)
    if mse < 1e-10:
        return torch.tensor(100.0, device=pred.device)
    return 10.0 * torch.log10(max_val ** 2 / mse)


def ssim_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """SSIM averaged per-frame over T.

    pred, gt: (B, 1, T, H, W).
    """
    try:
        from pytorch_msssim import ssim
        B, C, T, H, W = pred.shape
        total = torch.tensor(0.0, device=pred.device)
        for t in range(T):
            total = total + ssim(pred[:, :, t], gt[:, :, t], data_range=1.0, size_average=True)
        return total / T
    except ImportError:
        pass

    # Fallback manual SSIM
    B, C, T, H, W = pred.shape
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device, dtype=pred.dtype)
    kernel = kernel / kernel.numel()

    total = torch.tensor(0.0, device=pred.device)
    for t in range(T):
        p, g = pred[:, :, t], gt[:, :, t]
        mu_p = F.conv2d(p, kernel, padding=pad)
        mu_g = F.conv2d(g, kernel, padding=pad)
        sigma_pp = F.conv2d(p * p, kernel, padding=pad) - mu_p ** 2
        sigma_gg = F.conv2d(g * g, kernel, padding=pad) - mu_g ** 2
        sigma_pg = F.conv2d(p * g, kernel, padding=pad) - mu_p * mu_g
        ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)) / (
            (mu_p ** 2 + mu_g ** 2 + C1) * (sigma_pp + sigma_gg + C2)
        )
        total = total + ssim_map.mean()
    return total / T


def weighted_f_measure(
    pred: torch.Tensor,
    gt: torch.Tensor,
    beta: float = 0.3,
    num_thresholds: int = 255,
) -> torch.Tensor:
    """Weighted F-measure (Fβω) — threshold-sweep, max-F variant.

    Avoids hard 0.5 binarisation by sweeping thresholds and reporting max.

    Args:
        pred, gt: (B, 1, T, H, W) in [0, 1].
        beta: F-measure beta parameter.
        num_thresholds: Number of thresholds to sweep.
    Returns:
        Scalar max weighted F-measure.
    """
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    # Weight map: emphasise pixels near GT edges
    gt_mean = gt_flat.mean()
    weight = 1.0 + 5.0 * (gt_flat - gt_mean).abs()

    best_f = torch.tensor(0.0, device=pred.device)
    beta_sq = beta ** 2

    for thr_idx in range(1, num_thresholds + 1):
        thr = thr_idx / (num_thresholds + 1)
        binary_pred = (pred_flat >= thr).float()

        tp = (weight * binary_pred * gt_flat).sum()
        weighted_pred_sum = (weight * binary_pred).sum()
        weighted_gt_sum = (weight * gt_flat).sum()

        precision = tp / (weighted_pred_sum + 1e-8)
        recall = tp / (weighted_gt_sum + 1e-8)

        f = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8)
        best_f = torch.maximum(best_f, f)

    return best_f


def temporal_stability_score(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Temporal Stability Score (TSS).

    TSS = mean(|diff(pred, t)| - |diff(gt, t)|)

    Negative → model is smoother than GT (good).
    Positive → model is jitterier than GT (bad).
    """
    if pred.shape[2] <= 1:
        return torch.tensor(0.0, device=pred.device)
    diff_pred = (pred[:, :, 1:] - pred[:, :, :-1]).abs().mean()
    diff_gt = (gt[:, :, 1:] - gt[:, :, :-1]).abs().mean()
    return diff_pred - diff_gt


def lpips_metric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """LPIPS (lower is better). Requires ``lpips`` package.

    Computes per-frame LPIPS and averages over T.
    """
    try:
        import lpips
    except ImportError:
        return torch.tensor(float("nan"), device=pred.device)

    B, C, T, H, W = pred.shape
    # LPIPS expects 3-channel input; replicate single channel
    if C == 1:
        pred_3ch = pred.repeat(1, 3, 1, 1, 1)
        gt_3ch = gt.repeat(1, 3, 1, 1, 1)
    else:
        pred_3ch, gt_3ch = pred, gt

    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(pred.device)
    loss_fn.eval()

    total = torch.tensor(0.0, device=pred.device)
    with torch.no_grad():
        for t in range(T):
            # LPIPS expects [-1, 1] range
            p = pred_3ch[:, :, t] * 2.0 - 1.0
            g = gt_3ch[:, :, t] * 2.0 - 1.0
            total = total + loss_fn(p, g).mean()
    return total / T


def noise_robustness_score(
    clean_mae: float,
    noisy_mae: float,
) -> float:
    """Noise Robustness Score (NRS).

    Measures the relative degradation in MAE when moving from clean to noisy
    evaluation.  Lower is better (model is more robust to noise).

    NRS = (noisy_MAE - clean_MAE) / max(clean_MAE, eps)

    Returns:
        Scalar NRS value.  0 = perfectly robust, >0 = degrades under noise.
    """
    eps = 1e-8
    return (noisy_mae - clean_mae) / max(clean_mae, eps)


# ---------------------------------------------------------------------------
# Metric Suite (all-in-one)
# ---------------------------------------------------------------------------

class MetricSuite:
    """Convenience wrapper that computes all metrics at once.

    Usage:
        suite = MetricSuite()
        results = suite(pred, gt)   # dict of scalar tensors
    """

    def __init__(self, compute_lpips: bool = False):
        self.compute_lpips = compute_lpips

    @torch.no_grad()
    def __call__(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {
            "mae": mae(pred, gt),
            "psnr": psnr(pred, gt),
            "ssim": ssim_metric(pred, gt),
            "weighted_f": weighted_f_measure(pred, gt, beta=0.3),
            "tss": temporal_stability_score(pred, gt),
        }
        if self.compute_lpips:
            results["lpips"] = lpips_metric(pred, gt)
        return results
