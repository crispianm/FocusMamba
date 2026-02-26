"""
Training losses package.
"""

from .scale_invariant import ScaleInvariantLogLoss
from .distillation import DistillationLoss
from .combined import CombinedLoss

__all__ = ["ScaleInvariantLogLoss", "DistillationLoss", "CombinedLoss"]
