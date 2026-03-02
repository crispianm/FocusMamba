"""
Combined Loss
==============

Weighted sum of all loss components: SI-log, distillation, gradient
smoothness, temporal consistency, and uncertainty NLL.

Weights are read from config.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .scale_invariant import ScaleInvariantLogLoss
from .distillation import DistillationLoss


class CombinedLoss(nn.Module):
    """Weighted combination of all depth estimation losses.

    Args:
        cfg: Loss config dict with keys:
            si_log_weight, gradient_weight, temporal_weight, uncertainty_nll_weight
        distillation_cfg: Distillation-specific config (teacher configs, etc.)
    """

    def __init__(self, cfg: Dict[str, Any], distillation_cfg: Optional[Dict] = None):
        super().__init__()

        self.si_log_weight = cfg.get("si_log_weight", 1.0)
        self.gradient_weight = cfg.get("gradient_weight", 0.0)
        self.temporal_weight = cfg.get("temporal_weight", 0.0)
        self.uncertainty_weight = cfg.get("uncertainty_nll_weight", 0.0)

        self.si_log = ScaleInvariantLogLoss()

        # Distillation loss (if teachers are configured)
        self.distillation = None
        if distillation_cfg and distillation_cfg.get("enabled", False):
            self.distillation = DistillationLoss(
                teacher_configs=distillation_cfg.get("teachers", []),
                confidence_weighted=distillation_cfg.get("confidence_weighted", True),
            )

        # Gradient and temporal losses (to be implemented)
        self._gradient_loss = None
        self._temporal_loss = None

        try:
            from .gradient import GradientSmoothnessLoss
            self._gradient_loss = GradientSmoothnessLoss()
        except (ImportError, NotImplementedError):
            pass

        try:
            from .temporal import TemporalConsistencyLoss
            self._temporal_loss = TemporalConsistencyLoss()
        except (ImportError, NotImplementedError):
            pass

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_depths: Optional[Dict[str, torch.Tensor]] = None,
        gt_depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            student_outputs: Dict with 'depth' (B,1,T,H,W) and optionally 'uncertainty'.
            teacher_depths:  Dict mapping teacher name → (B,1,T,H,W) depth.
            gt_depth:        (B,1,T,H,W) optional ground-truth metric depth.
            mask:            (B,1,T,H,W) optional validity mask.

        Returns:
            Dict with 'total' and individual loss components.
        """
        student_depth = student_outputs["depth"]
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=student_depth.device, requires_grad=True)

        # Auto-derive a validity mask from GT depth when none is supplied.
        # TartanAir encodes invalid/sky pixels as depth=0; excluding them prevents
        # sky pixels in teacher outputs (e.g. DepthPro ~10 000 m) from corrupting
        # the SI-log loss.  A mask provided by the caller always takes precedence.
        if mask is None and gt_depth is not None:
            mask = (gt_depth > 0).float()

        # 1. Distillation loss (primary training signal)
        if self.distillation is not None and teacher_depths:
            distill_losses = self.distillation(student_depth, teacher_depths, mask)
            distill_total = distill_losses.get("total", torch.tensor(0.0))
            total = total + self.si_log_weight * distill_total
            losses.update(distill_losses)

        # 2. Direct SI-log loss against GT (if available)
        elif gt_depth is not None:
            si_loss = self.si_log(student_depth, gt_depth, mask)
            total = total + self.si_log_weight * si_loss
            losses["si_log"] = si_loss.detach()

        # 3. Gradient smoothness loss (if implemented)
        if self._gradient_loss is not None and self.gradient_weight > 0:
            target = gt_depth if gt_depth is not None else list(teacher_depths.values())[0]
            try:
                grad_loss = self._gradient_loss(student_depth, target, mask)
                total = total + self.gradient_weight * grad_loss
                losses["gradient"] = grad_loss.detach()
            except NotImplementedError:
                pass

        # 4. Temporal consistency loss (if implemented)
        if self._temporal_loss is not None and self.temporal_weight > 0:
            target = gt_depth if gt_depth is not None else list(teacher_depths.values())[0]
            try:
                temp_loss = self._temporal_loss(student_depth, target, mask)
                total = total + self.temporal_weight * temp_loss
                losses["temporal"] = temp_loss.detach()
            except NotImplementedError:
                pass

        # 5. Uncertainty NLL loss
        if self.uncertainty_weight > 0 and "uncertainty" in student_outputs:
            target = gt_depth if gt_depth is not None else (
                list(teacher_depths.values())[0] if teacher_depths else None
            )
            if target is not None:
                nll_loss = self._uncertainty_nll(
                    student_depth, student_outputs["uncertainty"], target
                )
                total = total + self.uncertainty_weight * nll_loss
                losses["uncertainty_nll"] = nll_loss.detach()

        losses["total"] = total
        return losses

    @staticmethod
    def _uncertainty_nll(
        pred: torch.Tensor,
        log_variance: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian NLL loss for uncertainty estimation.

        loss = 0.5 * (log_var + (pred - target)² / exp(log_var))
        """
        precision = torch.exp(-log_variance)
        sq_err = (torch.log(pred.clamp(min=1e-6)) - torch.log(target.clamp(min=1e-6))) ** 2
        nll = 0.5 * (log_variance + precision * sq_err)
        return nll.mean()
