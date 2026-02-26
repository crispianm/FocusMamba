"""
iPhone LiDAR Evaluation Protocol
==================================

Ground-truth evaluation using paired RGB+LiDAR data captured with iPhone.
This provides the only real metric ground truth in the benchmark.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def run_iphone_eval(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on iPhone LiDAR ground-truth dataset.

    Returns:
        Dict of depth metrics.
    """
    raise NotImplementedError(
        "Implement: load iPhone LiDAR dataset, run model, "
        "compute_depth_metrics against LiDAR ground truth."
    )
