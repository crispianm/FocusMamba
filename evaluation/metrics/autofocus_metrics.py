"""
Autofocus Metrics
==================

Focus accuracy (distance error), focus latency, pull success rate.

These metrics evaluate how well predicted depth translates to correct
focus distance commands for cinematic autofocus.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def focus_distance_error(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    bbox: Tuple[int, int, int, int],
) -> float:
    """Compute focus distance error for a subject bounding box.

    Takes median depth within the subject ROI as the focus distance.

    Args:
        pred_depth: (H, W) predicted depth map.
        gt_depth:   (H, W) ground-truth depth map.
        bbox: (x1, y1, x2, y2) bounding box of the subject.

    Returns:
        Absolute focus distance error in metres.
    """
    x1, y1, x2, y2 = bbox
    pred_roi = pred_depth[y1:y2, x1:x2]
    gt_roi = gt_depth[y1:y2, x1:x2]

    pred_focus = pred_roi.median().item()
    gt_focus = gt_roi.median().item()

    return abs(pred_focus - gt_focus)


def focus_pull_success_rate(
    distance_errors: List[float],
    threshold_m: float = 0.1,
) -> float:
    """Fraction of frames where focus distance error < threshold.

    Args:
        distance_errors: List of per-frame focus distance errors (metres).
        threshold_m: Acceptable error threshold in metres.

    Returns:
        Success rate in [0, 1].
    """
    if not distance_errors:
        return 0.0
    successes = sum(1 for e in distance_errors if e < threshold_m)
    return successes / len(distance_errors)


def focus_latency(
    pred_depths_over_time: List[float],
    gt_depths_over_time: List[float],
    threshold_m: float = 0.1,
    fps: float = 30.0,
) -> Optional[float]:
    """Measure how many milliseconds it takes for the predicted focus to
    converge within threshold of ground truth after a subject depth change.

    Args:
        pred_depths_over_time: List of predicted focus distances per frame.
        gt_depths_over_time:   List of GT focus distances per frame.
        threshold_m: Convergence threshold in metres.
        fps: Frame rate for time conversion.

    Returns:
        Convergence latency in milliseconds, or None if never converges.
    """
    for i, (p, g) in enumerate(zip(pred_depths_over_time, gt_depths_over_time)):
        if abs(p - g) < threshold_m:
            return i * (1000.0 / fps)
    return None
