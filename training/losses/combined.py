"""
Combined Loss
==============

Video Depth Anything-style objective with optional distillation.

Total loss:
    L = beta * L_ssi + alpha * L_tgm + gamma * L_distill

Supports two explicit training targets:
- metric:   supervised metric depth + metric distillation
- relative: supervised relative depth + relative distillation
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .scale_shift import ScaleShiftInvariantLoss
from .distillation import DistillationLoss


class CombinedLoss(nn.Module):
    """Weighted combination of SSI + TGM (+ distillation)."""

    def __init__(self, cfg: Dict[str, Any], distillation_cfg: Optional[Dict] = None):
        super().__init__()

        self.ssi_weight = float(cfg.get("ssi_weight", cfg.get("si_log_weight", 1.0)))
        self.temporal_weight = float(cfg.get("temporal_weight", 10.0))
        self.distillation_weight = cfg.get("distillation_weight", 1.0)
        self.target_mode = str(cfg.get("training_target", "metric")).lower()
        if self.target_mode not in ("metric", "relative"):
            raise ValueError(f"Unknown loss.training_target={self.target_mode!r}")

        self.relative_loss_type = str(cfg.get("relative_loss_type", "l1")).lower()
        self.relative_log_space = bool(cfg.get("relative_log_space", True))
        self.relative_q_low = float(cfg.get("relative_q_low", 0.02))
        self.relative_q_high = float(cfg.get("relative_q_high", 0.98))
        self.relative_clamp = bool(cfg.get("relative_clamp", True))
        self.eps = 1e-6

        self.ssi = ScaleShiftInvariantLoss(mode=cfg.get("ssi_mode", "l2"))

        self.distillation = None
        if distillation_cfg and distillation_cfg.get("enabled", False):
            relative_cfg = distillation_cfg.get("relative", None)
            if relative_cfg is None:
                relative_cfg = {
                    "log_space": self.relative_log_space,
                    "q_low": self.relative_q_low,
                    "q_high": self.relative_q_high,
                    "clamp": self.relative_clamp,
                }
            self.distillation = DistillationLoss(
                teacher_configs=distillation_cfg.get("teachers", []),
                confidence_weighted=distillation_cfg.get("confidence_weighted", True),
                lambda_si=float(distillation_cfg.get("lambda_si", 0.5)),
                mtkd_cfg=distillation_cfg.get("mtkd", {}),
                target_mode=distillation_cfg.get("target_mode", self.target_mode),
                relative_cfg=relative_cfg,
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

    def _normalize_relative_depth(
        self,
        depth: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.relative_log_space:
            base = torch.log(depth.clamp(min=self.eps))
        else:
            base = depth

        valid = torch.isfinite(base)
        if mask is not None:
            valid = valid & mask.bool()

        norm = torch.zeros_like(base)
        b = base.shape[0]
        for i in range(b):
            vi = valid[i]
            if vi.sum() < 8:
                continue
            vals = base[i][vi].detach()
            q_low = torch.quantile(vals, self.relative_q_low)
            q_high = torch.quantile(vals, self.relative_q_high)
            denom = (q_high - q_low).clamp(min=self.eps)
            x = (base[i] - q_low) / denom
            if self.relative_clamp:
                x = x.clamp(0.0, 1.0)
            norm[i] = torch.where(vi, x, torch.zeros_like(x))
        return norm

    @staticmethod
    def _masked_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = (pred - target).abs()
        if mask is not None:
            valid = mask.bool() & torch.isfinite(diff)
            if valid.any():
                return diff[valid].mean()
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        finite = torch.isfinite(diff)
        if finite.any():
            return diff[finite].mean()
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_depths: Optional[Dict[str, torch.Tensor]] = None,
        gt_depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        student_depth = student_outputs["depth"]
        student_relative = student_outputs.get("depth_relative", student_depth)
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=student_depth.device, requires_grad=True)

        if mask is None and gt_depth is not None:
            mask = (gt_depth > 0).float()

        # Distillation.
        if self.distillation is not None and teacher_depths:
            distill_pred = student_relative if self.target_mode == "relative" else student_depth
            distill_losses = self.distillation(distill_pred, teacher_depths, mask)
            distill_total = distill_losses.get("total", torch.tensor(0.0, device=student_depth.device))
            total = total + (self.distillation_weight * distill_total)
            losses.update(distill_losses)

        # GT supervision.
        if gt_depth is not None:
            if self.target_mode == "relative":
                gt_rel = self._normalize_relative_depth(gt_depth, mask)
                pred_rel = student_relative

                if self.relative_loss_type == "ssi":
                    aligned_pred = self.ssi.align_scale_shift(pred_rel, gt_rel, mask=mask)
                    ssi_loss = self.ssi(pred_rel, gt_rel, mask=mask, aligned_pred=aligned_pred)
                else:
                    ssi_loss = self._masked_l1(pred_rel, gt_rel, mask=mask)
                    aligned_pred = pred_rel
                total = total + (self.ssi_weight * ssi_loss)
                losses["ssi"] = ssi_loss.detach()

                if self._temporal_loss is not None and self.temporal_weight > 0:
                    try:
                        tgm_loss = self._temporal_loss(aligned_pred, gt_rel, mask)
                        total = total + (self.temporal_weight * tgm_loss)
                        losses["tgm"] = tgm_loss.detach()
                    except NotImplementedError:
                        pass
            else:
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
