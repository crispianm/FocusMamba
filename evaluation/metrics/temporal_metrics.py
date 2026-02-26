"""
Temporal Metrics
=================

Temporal consistency error (OPW), flicker score.

Stub — user implements optical-flow-warping consistency.
"""

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


def flicker_score(depth_sequence: torch.Tensor) -> float:
    """Compute depth flicker score — mean absolute frame-to-frame depth change.

    Args:
        depth_sequence: (B, 1, T, H, W) depth predictions.

    Returns:
        Mean flicker magnitude.
    """
    # Simple proxy: mean absolute difference between consecutive frames
    if depth_sequence.shape[2] < 2:
        return 0.0
    diffs = (depth_sequence[:, :, 1:] - depth_sequence[:, :, :-1]).abs()
    return diffs.mean().item()
