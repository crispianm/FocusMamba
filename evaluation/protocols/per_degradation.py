"""
Per-Degradation Protocol
=========================

Sweep over all degradation types at multiple severity levels.
Generates the data for paper Figure 3/4 (performance vs. severity curves).
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def run_degradation_sweep(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model under each degradation type at 3 severity levels.

    Returns:
        Nested dict: {degradation_type: {severity: {metric: value}}}.
    """
    raise NotImplementedError(
        "Implement: for each degradation type in [noise, blur, atmospheric, compression, lens], "
        "apply at severity=[low, medium, high], run model, compute metrics."
    )
