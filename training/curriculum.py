"""
Degradation Severity Curriculum Scheduler
==========================================

Increases degradation severity over training epochs, starting with
mild degradations and ramping up to full severity.

Interfaces with the DegradationPipeline in dataloader/degradation/pipeline.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict


class CurriculumScheduler:
    """Curriculum scheduler for degradation severity.

    Gradually increases the severity scale from 0 → 1 over training.

    Args:
        warmup_epochs: Number of epochs with minimal degradation.
        max_severity_epoch: Epoch at which full severity is reached.
        schedule: 'linear' or 'cosine'.
    """

    def __init__(
        self,
        warmup_epochs: int = 10,
        max_severity_epoch: int = 100,
        schedule: str = "linear",
    ):
        self.warmup_epochs = warmup_epochs
        self.max_severity_epoch = max_severity_epoch
        self.schedule = schedule

    def get_scale(self, epoch: int) -> float:
        """Get the current severity scale for a given epoch.

        Returns:
            float in [0, 1]. 0 = no degradation, 1 = full severity range.
        """
        if epoch < self.warmup_epochs:
            # During warmup, scale linearly from 0 to a small value
            return 0.1 * (epoch / max(1, self.warmup_epochs))

        # After warmup, ramp to 1.0
        progress = (epoch - self.warmup_epochs) / max(
            1, self.max_severity_epoch - self.warmup_epochs
        )
        progress = min(1.0, progress)

        if self.schedule == "cosine":
            # Cosine schedule: smooth ramp
            scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            # Linear schedule
            scale = progress

        # Map from [0, 1] progress to [0.1, 1.0] severity
        return 0.1 + 0.9 * scale

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "CurriculumScheduler":
        """Create from config dict."""
        deg_cfg = cfg.get("degradation", {})
        curriculum_cfg = deg_cfg.get("curriculum", {})

        if isinstance(curriculum_cfg, bool):
            # Handle case where curriculum is just a bool flag
            if not curriculum_cfg:
                return cls(warmup_epochs=0, max_severity_epoch=1)
            return cls()

        return cls(
            warmup_epochs=curriculum_cfg.get("warmup_epochs", 10),
            max_severity_epoch=curriculum_cfg.get("max_severity_epoch", 100),
            schedule=curriculum_cfg.get("schedule", "linear"),
        )
