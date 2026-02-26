"""
Multi-Teacher Distillation Loss
================================

Core training loss for the degradation-robust depth estimation model.
This is the most novel training component.

For each training sample:
    1. Run student on DEGRADED clip   → student_depth
    2. Run each teacher on CLEAN clip → teacher_depths (no_grad)
    3. Compute per-teacher loss:
        image_teacher_loss  = SI_log(student_depth, image_teacher_depth)
        video_teacher_loss  = SI_log(student_depth, video_teacher_depth)
                            + temporal_consistency(student_depth, video_teacher_depth)
    4. Ensemble teacher targets via fixed or learned weighting
    5. Optional: confidence-weighted loss (down-weight where teachers disagree)

Key design decision: teachers are run on the *clean* version of the clip.
The student must learn to produce clean-equivalent depth from degraded input.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .scale_invariant import ScaleInvariantLogLoss


class DistillationLoss(nn.Module):
    """Multi-teacher distillation loss.

    Args:
        teacher_configs: List of dicts with keys:
            'name': str, 'weight': float, 'loss': str ('si_log' or 'si_log_plus_temporal')
        confidence_weighted: If True, down-weight loss where teachers disagree.
        lambda_si: SI-log lambda parameter.
    """

    def __init__(
        self,
        teacher_configs: List[Dict],
        confidence_weighted: bool = True,
        lambda_si: float = 0.5,
    ):
        super().__init__()
        self.teacher_configs = teacher_configs
        self.confidence_weighted = confidence_weighted
        self.si_log = ScaleInvariantLogLoss(lambda_si=lambda_si)

    def forward(
        self,
        student_depth: torch.Tensor,
        teacher_depths: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-teacher distillation loss.

        Args:
            student_depth:  (B, 1, T, H, W) student's predicted metric depth.
            teacher_depths: Dict mapping teacher name → (B, 1, T, H, W) teacher depth.
            mask:           (B, 1, T, H, W) optional validity mask.

        Returns:
            Dict with 'total', per-teacher losses, and optionally 'confidence_weight'.
        """
        losses = {}
        weighted_sum = torch.tensor(0.0, device=student_depth.device)
        total_weight = 0.0

        # Compute per-teacher losses
        for cfg in self.teacher_configs:
            name = cfg["name"]
            weight = cfg["weight"]
            loss_type = cfg.get("loss", "si_log")

            if name not in teacher_depths:
                continue

            teacher_d = teacher_depths[name]

            # SI-log loss
            si_loss = self.si_log(student_depth, teacher_d, mask)
            losses[f"distill/{name}_si_log"] = si_loss.detach()

            teacher_loss = si_loss

            # Add temporal consistency for video teachers
            if loss_type == "si_log_plus_temporal" and student_depth.shape[2] > 1:
                temporal_loss = self._temporal_consistency(student_depth, teacher_d)
                losses[f"distill/{name}_temporal"] = temporal_loss.detach()
                teacher_loss = teacher_loss + 0.5 * temporal_loss

            weighted_sum = weighted_sum + weight * teacher_loss
            total_weight += weight

        # Apply confidence weighting (downweight where teachers disagree)
        if self.confidence_weighted and len(teacher_depths) > 1:
            confidence = self._compute_confidence(teacher_depths)
            # confidence shape: (B, 1, T, H, W), values in [0, 1]
            # Not applied to the loss directly yet — available for future use
            losses["confidence_mean"] = confidence.mean().detach()

        if total_weight > 0:
            losses["total"] = weighted_sum / total_weight
        else:
            losses["total"] = torch.tensor(0.0, device=student_depth.device, requires_grad=True)

        return losses

    @staticmethod
    def _temporal_consistency(
        pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Temporal consistency loss: L1 on temporal gradients.

        Args:
            pred:   (B, 1, T, H, W)
            target: (B, 1, T, H, W)
        """
        if pred.shape[2] <= 1:
            return torch.tensor(0.0, device=pred.device)

        log_pred = torch.log(pred.clamp(min=1e-6))
        log_target = torch.log(target.clamp(min=1e-6))

        td_pred = log_pred[:, :, 1:] - log_pred[:, :, :-1]
        td_target = log_target[:, :, 1:] - log_target[:, :, :-1]
        return (td_pred - td_target).abs().mean()

    @staticmethod
    def _compute_confidence(
        teacher_depths: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute confidence map from teacher ensemble agreement.

        Low confidence where teachers disagree (high variance).

        Returns:
            confidence: (B, 1, T, H, W) in [0, 1].
        """
        depths = list(teacher_depths.values())
        if len(depths) < 2:
            return torch.ones_like(depths[0])

        # Stack and compute log-depth variance
        stacked = torch.stack([torch.log(d.clamp(min=1e-6)) for d in depths], dim=0)
        variance = stacked.var(dim=0)  # (B, 1, T, H, W)

        # Convert variance to confidence: high variance → low confidence
        # Using exponential decay: conf = exp(-k * var)
        confidence = torch.exp(-5.0 * variance)
        return confidence
