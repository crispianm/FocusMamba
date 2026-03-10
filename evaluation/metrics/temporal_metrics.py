"""Temporal metrics for streaming depth evaluation."""

from __future__ import annotations

import torch


def temporal_consistency_error(
    depth_sequence: torch.Tensor,
    flow_forward: torch.Tensor,
) -> float:
    """Compute optical-flow-warping temporal consistency error.

    Warp depth[t] to depth[t+1] via optical flow and measure agreement.

    Args:
        depth_sequence: (B, 1, T, H, W) predicted depth over T frames.
        flow_forward:   (B, 2, T-1, H, W) forward optical flow from t→t+1.

    Returns:
        Mean warping error.
    """
    raise NotImplementedError(
        "Implement OPW-based temporal consistency. "
        "Use kornia.geometry.transform.warp or cv2.remap."
    )


def frame_depth_variation(
    depth_sequence: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> float:
    """Compute normalized frame-to-frame depth variation.

    Args:
        depth_sequence: (B, 1, T, H, W) predicted depth.
        mask: Optional validity mask with the same shape.
        eps: Numerical floor for normalization.

    Returns:
        Mean absolute consecutive prediction difference normalized by the
        sequence mean predicted depth, yielding a dimensionless temporal score.
    """
    if depth_sequence.shape[2] < 2:
        return 0.0
    depth = depth_sequence.float()
    diffs = (depth[:, :, 1:] - depth[:, :, :-1]).abs()

    if mask is not None:
        pair_mask = mask[:, :, 1:].bool() & mask[:, :, :-1].bool()
        if pair_mask.any():
            diff_mean = diffs[pair_mask].mean()
            mean_depth = depth[mask.bool()].abs().mean() if mask.bool().any() else depth.abs().mean()
        else:
            return 0.0
    else:
        diff_mean = diffs.mean()
        mean_depth = depth.abs().mean()

    mean_depth = mean_depth.clamp_min(eps)
    return (diff_mean / mean_depth).item()


def flicker_score(depth_sequence: torch.Tensor) -> float:
    """Backward-compatible alias for the formal FDV metric."""
    return frame_depth_variation(depth_sequence)
