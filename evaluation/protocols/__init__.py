"""Evaluation protocols package."""

from .per_degradation import run_degradation_sweep
from .robodepth import run_robodepth_eval

__all__ = [
    "run_degradation_sweep",
    "run_robodepth_eval",
]
