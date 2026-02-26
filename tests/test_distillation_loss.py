"""
Test Distillation Loss
========================

Loss should go to 0 when student depth == teacher depth.
"""

from __future__ import annotations

import torch
import pytest


class TestDistillationLoss:
    """Tests for the distillation loss."""

    def test_zero_loss_when_equal(self):
        """Loss should be 0 when student matches teacher exactly."""
        from training.losses.distillation import DistillationLoss

        loss_fn = DistillationLoss(teacher_names=["teacher_a"])

        student_depth = torch.ones(2, 1, 4, 64, 64)
        teacher_depths = {"teacher_a": torch.ones(2, 1, 4, 64, 64)}

        loss = loss_fn(student_depth, teacher_depths)
        assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"

    def test_positive_loss_when_different(self):
        """Loss should be positive when student ≠ teacher."""
        from training.losses.distillation import DistillationLoss

        loss_fn = DistillationLoss(teacher_names=["teacher_a"])

        student_depth = torch.ones(2, 1, 4, 64, 64)
        teacher_depths = {"teacher_a": torch.ones(2, 1, 4, 64, 64) * 2.0}

        loss = loss_fn(student_depth, teacher_depths)
        assert loss.item() > 0

    def test_multi_teacher(self):
        """Loss should handle multiple teachers."""
        from training.losses.distillation import DistillationLoss

        loss_fn = DistillationLoss(teacher_names=["t1", "t2", "t3"])

        student_depth = torch.rand(2, 1, 4, 64, 64).clamp(0.01)
        teacher_depths = {
            "t1": torch.rand(2, 1, 4, 64, 64).clamp(0.01),
            "t2": torch.rand(2, 1, 4, 64, 64).clamp(0.01),
            "t3": torch.rand(2, 1, 4, 64, 64).clamp(0.01),
        }

        loss = loss_fn(student_depth, teacher_depths)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_confidence_weighting(self):
        """With use_confidence=True, loss should still work."""
        from training.losses.distillation import DistillationLoss

        loss_fn = DistillationLoss(
            teacher_names=["t1", "t2"],
            use_confidence_weighting=True,
        )

        student_depth = torch.rand(2, 1, 4, 64, 64).clamp(0.01)
        teacher_depths = {
            "t1": torch.rand(2, 1, 4, 64, 64).clamp(0.01),
            "t2": torch.rand(2, 1, 4, 64, 64).clamp(0.01),
        }

        loss = loss_fn(student_depth, teacher_depths)
        assert torch.isfinite(loss)
