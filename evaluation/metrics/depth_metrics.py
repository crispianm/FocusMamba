"""
Depth Metrics
==============

Standard depth estimation metrics: AbsRel, SqRel, RMSE, δ1/δ2/δ3, SI-log.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


@torch.no_grad()
def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> Dict[str, float]:
    """Compute standard monocular depth estimation metrics.

    Args:
        pred: (B, 1, T, H, W) or (B, 1, H, W) predicted metric depth.
        gt:   Same shape, ground-truth metric depth.
        mask: Same shape, optional valid pixel mask.
        min_depth: Minimum valid depth (metres).
        max_depth: Maximum valid depth (metres).

    Returns:
        Dict with: abs_rel, sq_rel, rmse, rmse_log, si_log,
                    delta1, delta2, delta3.
    """
    pred = pred.float().flatten()
    gt = gt.float().flatten()
    if mask is not None:
        mask = mask.flatten().bool()
    else:
        mask = torch.ones_like(gt, dtype=torch.bool)

    # Apply depth range + validity
    valid = mask & (gt > min_depth) & (gt < max_depth) & (pred > min_depth) & torch.isfinite(pred)
    pred = pred[valid]
    gt = gt[valid]

    if pred.numel() == 0:
        return {k: float("nan") for k in [
            "abs_rel", "sq_rel", "rmse", "rmse_log", "si_log",
            "delta1", "delta2", "delta3",
        ]}

    diff = pred - gt
    abs_diff = diff.abs()

    # AbsRel
    abs_rel = (abs_diff / gt).mean().item()

    # SqRel
    sq_rel = ((diff ** 2) / gt).mean().item()

    # RMSE
    rmse = (diff ** 2).mean().sqrt().item()

    # RMSE log
    log_diff = (torch.log(pred) - torch.log(gt))
    rmse_log = (log_diff ** 2).mean().sqrt().item()

    # SI-log  (scale-invariant logarithmic error)
    si_log = (log_diff ** 2).mean().item() - (log_diff.mean().item()) ** 2

    # δ thresholds
    ratio = torch.max(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "si_log": si_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }
