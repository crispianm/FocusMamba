"""
Clean Baseline Protocol
========================

Performance on clean (undegraded) input — serves as a regression check
to ensure degradation training hasn't hurt clean-input performance.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def run_clean_baseline(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on clean validation set.

    Returns:
        Dict of depth metrics (abs_rel, delta1, etc.).
    """
    raise NotImplementedError(
        "Implement: load clean validation set, run model, compute_depth_metrics."
    )
