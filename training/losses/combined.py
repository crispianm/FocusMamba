"""
Combined Loss
==============

Video Depth Anything-style objective with optional distillation novelty.

Total loss:
    L = beta * L_ssi + alpha * L_tgm + gamma * L_distill

Weights are read from config.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .scale_shift import ScaleShiftInvariantLoss
from .distillation import DistillationLoss


class CombinedLoss(nn.Module):
    """Weighted combination of SSI + TGM (+ distillation).

    Args:
        cfg: Loss config dict with keys:
            ssi_weight, temporal_weight, temporal_threshold, temporal_log_space,
            distillation_weight
        distillation_cfg: Distillation-specific config (teacher configs, etc.)
    """

    def __init__(self, cfg: Dict[str, Any], distillation_cfg: Optional[Dict] = None):
        super().__init__()

        # VDA Eq.(4): L_all = alpha * L_TGM + beta * L_ssi
        # Keep backward-compatibility with older config names.
        self.ssi_weight = float(cfg.get("ssi_weight", cfg.get("si_log_weight", 1.0)))
        self.temporal_weight = float(cfg.get("temporal_weight", 10.0))
        self.distillation_weight = cfg.get("distillation_weight", 1.0)

        self.ssi = ScaleShiftInvariantLoss(mode=cfg.get("ssi_mode", "l2"))

        # Distillation loss (if teachers are configured)
        self.distillation = None
        if distillation_cfg and distillation_cfg.get("enabled", False):
            self.distillation = DistillationLoss(
                teacher_configs=distillation_cfg.get("teachers", []),
                confidence_weighted=distillation_cfg.get("confidence_weighted", True),
            )

        self._temporal_loss = None
        try:
            from .temporal import TemporalConsistencyLoss
            self._temporal_loss = TemporalConsistencyLoss(
                threshold=cfg.get("temporal_threshold", 0.05),
                log_space=cfg.get("temporal_log_space", False),
            )
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

        # 1. Distillation loss (from teacher pseudo-labels)
        if self.distillation is not None and teacher_depths:
            distill_losses = self.distillation(student_depth, teacher_depths, mask)
            distill_total = distill_losses.get("total", torch.tensor(0.0))
            total = total + (self.distillation_weight * distill_total)
            losses.update(distill_losses)

        # 2. VDA-style GT supervision: SSI + TGM
        if gt_depth is not None:
            aligned_pred = self.ssi.align_scale_shift(student_depth, gt_depth, mask=mask)
            ssi_loss = self.ssi(student_depth, gt_depth, mask=mask, aligned_pred=aligned_pred)
            total = total + (self.ssi_weight * ssi_loss)
            losses["ssi"] = ssi_loss.detach()

            if self._temporal_loss is not None and self.temporal_weight > 0:
                try:
                    tgm_loss = self._temporal_loss(aligned_pred, gt_depth, mask)
                    total = total + (self.temporal_weight * tgm_loss)
                    losses["tgm"] = tgm_loss.detach()
                except NotImplementedError:
                    pass

        losses["total"] = total
        return losses
