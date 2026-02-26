"""
Baseline: Run Depth Anything
==============================

Run Depth Anything V3 as a standalone baseline for comparison.
"""

from __future__ import annotations

from typing import Any, Dict


def run_depth_anything_baseline(config: Dict[str, Any]) -> Dict[str, float]:
    """Run Depth Anything V3 on the evaluation set.

    Returns:
        Dict of depth metrics.
    """
    raise NotImplementedError(
        "Implement: load Depth Anything V3, run on eval set, compute metrics."
    )
