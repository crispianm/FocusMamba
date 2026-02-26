"""
Training package for degradation-robust metric video depth estimation.

Provides:
    - losses/       : SI-log, distillation, gradient, temporal, combined
    - trainer.py    : Full training/validation loop
    - curriculum.py : Degradation severity scheduling
    - ema.py        : Exponential moving average of student weights
    - callbacks/    : TensorBoard visualisation, benchmarking, profiling
"""

from .losses import ScaleInvariantLogLoss, DistillationLoss, CombinedLoss
from .curriculum import CurriculumScheduler
from .ema import EMAModel
from .trainer import WarmupCosineScheduler, train_one_epoch, validate, compute_depth_metrics

__all__ = [
    "ScaleInvariantLogLoss",
    "DistillationLoss",
    "CombinedLoss",
    "CurriculumScheduler",
    "EMAModel",
    "WarmupCosineScheduler",
    "train_one_epoch",
    "validate",
    "compute_depth_metrics",
]
