"""
Training losses package.
"""

from .scale_invariant import ScaleInvariantLogLoss
from .distillation import DistillationLoss
from .combined import CombinedLoss
from .temporal import TemporalConsistencyLoss
from .gradient import GradientSmoothnessLoss

__all__ = [
    "ScaleInvariantLogLoss",
    "DistillationLoss",
    "CombinedLoss",
    "TemporalConsistencyLoss",
    "GradientSmoothnessLoss",
]