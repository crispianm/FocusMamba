"""
Baseline: Run UniDepth
========================

Run UniDepth as a standalone baseline for comparison.
"""

from __future__ import annotations

from typing import Any, Dict


def run_unidepth_baseline(config: Dict[str, Any]) -> Dict[str, float]:
    """Run UniDepth on the evaluation set.

    Returns:
        Dict of depth metrics.
    """
    raise NotImplementedError("Implement: load UniDepth, run on eval set, compute metrics.")
