"""
Baseline: Frame-wise Robust
==============================

Apply an image-based depth model frame-by-frame (no temporal context).
This is the key ablation — showing the benefit of temporal Mamba processing.
"""

from __future__ import annotations

from typing import Any, Dict


def run_framewise_robust_baseline(config: Dict[str, Any]) -> Dict[str, float]:
    """Run a per-frame depth model on the evaluation set.

    This strips temporal context and measures the degradation in
    temporal consistency, demonstrating Mamba's temporal advantage.

    Returns:
        Dict of depth + temporal metrics.
    """
    raise NotImplementedError(
        "Implement: load image depth model, apply per-frame, "
        "compute depth + temporal metrics."
    )
