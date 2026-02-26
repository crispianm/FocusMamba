"""
Baseline: Run Metric3D v2
===========================

Run Metric3D v2 as a standalone baseline for comparison.
"""

from __future__ import annotations

from typing import Any, Dict


def run_metric3d_baseline(config: Dict[str, Any]) -> Dict[str, float]:
    """Run Metric3D v2 on the evaluation set.

    Returns:
        Dict of depth metrics.
    """
    raise NotImplementedError("Implement: load Metric3D v2, run on eval set, compute metrics.")
