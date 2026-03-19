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
import torch.nn.functional as F

from .scale_shift import ScaleShiftInvariantLoss
from .distillation import DistillationLoss
from .gradient import GradientSmoothnessLoss


class CombinedLoss(nn.Module):
    """Weighted combination of SSI + TGM (+ distillation)."""

    def __init__(self, cfg: Dict[str, Any], distillation_cfg: Optional[Dict] = None):
        super().__init__()

        self.ssi_weight = float(cfg.get("ssi_weight", cfg.get("si_log_weight", 1.0)))
        self.temporal_weight = float(cfg.get("temporal_weight", 10.0))
        self.distillation_weight = cfg.get("distillation_weight", 1.0)
        self.gradient_weight = float(cfg.get("gradient_weight", 0.0))
        # Optional absolute metric anchors (useful for Stage-B conversion).
        self.metric_l1_weight = float(cfg.get("metric_l1_weight", 0.0))
        self.metric_log_l1_weight = float(cfg.get("metric_log_l1_weight", 0.0))
        self.boundary_weight = float(cfg.get("boundary_weight", 0.0))
        self.boundary_threshold = float(cfg.get("boundary_threshold", 0.05))
        self.boundary_dilation = int(cfg.get("boundary_dilation", 1))
        self.range_prior_weight = float(cfg.get("range_prior_weight", 0.0))
        self.min_depth = float(cfg.get("min_depth", 1e-3))
        self.max_depth = float(cfg.get("max_depth", 80.0))
        self.weight_ramp_cfg = cfg.get("weight_ramp", {}) or {}
        auxiliary_cfg = cfg.get("auxiliary", {}) or {}
        self.feature_alignment_weight = float(
            auxiliary_cfg.get("feature_alignment_weight", 0.0)
        )
        self.clean_depth_consistency_weight = float(
            auxiliary_cfg.get("clean_depth_consistency_weight", 0.0)
        )
        self.clean_depth_consistency_mode = str(
            auxiliary_cfg.get("clean_depth_consistency_mode", "log_l1")
        ).lower()
        self.edge_loss_weight = float(auxiliary_cfg.get("edge_loss_weight", 0.0))
        self.recon_l1_weight = float(auxiliary_cfg.get("recon_l1_weight", 0.0))
        self.recon_ssim_weight = float(auxiliary_cfg.get("recon_ssim_weight", 0.0))
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
        self.gradient = GradientSmoothnessLoss(
            mode=str(cfg.get("gradient_mode", "matching")).lower(),
            scales=int(cfg.get("gradient_scales", 4)),
            log_space=bool(cfg.get("gradient_log_space", True)),
        )
        self._progress = {
            "epoch": 0.0,
            "global_step": 0.0,
            "max_epochs": 1.0,
            "total_steps": 1.0,
        }

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
                distillation_cfg=distillation_cfg,
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

    def set_progress(
        self,
        *,
        epoch: int | float | None = None,
        global_step: int | float | None = None,
        max_epochs: int | float | None = None,
        total_steps: int | float | None = None,
    ) -> None:
        if epoch is not None:
            self._progress["epoch"] = float(epoch)
        if global_step is not None:
            self._progress["global_step"] = float(global_step)
        if max_epochs is not None:
            self._progress["max_epochs"] = max(float(max_epochs), 1.0)
        if total_steps is not None:
            self._progress["total_steps"] = max(float(total_steps), 1.0)

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

    def _current_weight(self, name: str, default: float) -> float:
        ramp = self.weight_ramp_cfg.get(name)
        if not isinstance(ramp, dict):
            return float(default)

        domain = str(ramp.get("domain", "epoch")).lower()
        if domain == "step":
            position = self._progress["global_step"]
            start = float(ramp.get("start_step", 0.0))
            end = float(ramp.get("end_step", self._progress["total_steps"]))
        else:
            position = self._progress["epoch"]
            start = float(ramp.get("start_epoch", 0.0))
            end = float(ramp.get("end_epoch", self._progress["max_epochs"]))

        start_value = float(ramp.get("start_value", 0.0))
        end_value = float(ramp.get("end_value", default))
        if end <= start:
            return end_value if position >= end else start_value
        if position <= start:
            return start_value
        if position >= end:
            return end_value
        alpha = (position - start) / max(end - start, self.eps)
        return start_value + (end_value - start_value) * alpha

    @staticmethod
    def _flatten_clip(x: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int]:
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), b, t, h, w

    def _boundary_mask(
        self,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        log_space: bool,
    ) -> torch.Tensor:
        base = torch.log(target.clamp(min=self.eps)) if log_space else target
        flat, b, t, h, w = self._flatten_clip(base)
        gx = flat[:, :, :, 1:] - flat[:, :, :, :-1]
        gy = flat[:, :, 1:, :] - flat[:, :, :-1, :]
        gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
        gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
        edge = torch.sqrt(gx.square() + gy.square()) > self.boundary_threshold

        dilation = max(int(self.boundary_dilation), 0)
        if dilation > 0:
            kernel = 2 * dilation + 1
            edge = (
                torch.nn.functional.max_pool2d(
                    edge.float(),
                    kernel_size=kernel,
                    stride=1,
                    padding=dilation,
                )
                > 0
            )

        edge = edge.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        if mask is not None:
            edge = edge & mask.bool()
        return edge

    def _range_prior(
        self,
        pred: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        penalty = (self.min_depth - pred).clamp(min=0.0) + (
            pred - self.max_depth
        ).clamp(min=0.0)
        if mask is not None:
            valid = mask.bool() & torch.isfinite(penalty)
            if valid.any():
                return penalty[valid].mean()
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        finite = torch.isfinite(penalty)
        if finite.any():
            return penalty[finite].mean()
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
        ssi_weight = self._current_weight("ssi_weight", self.ssi_weight)
        metric_l1_weight = self._current_weight(
            "metric_l1_weight", self.metric_l1_weight
        )
        metric_log_l1_weight = self._current_weight(
            "metric_log_l1_weight", self.metric_log_l1_weight
        )
        temporal_weight = self._current_weight("temporal_weight", self.temporal_weight)
        gradient_weight = self._current_weight("gradient_weight", self.gradient_weight)
        boundary_weight = self._current_weight("boundary_weight", self.boundary_weight)
        range_prior_weight = self._current_weight(
            "range_prior_weight", self.range_prior_weight
        )

        if mask is None and gt_depth is not None:
            mask = (gt_depth > 0).float()

        # Distillation.
        if self.distillation is not None and teacher_depths:
            distill_pred = (
                student_relative if self.target_mode == "relative" else student_depth
            )
            distill_losses = self.distillation(
                distill_pred,
                teacher_depths,
                mask=mask,
                gt_depth=gt_depth,
            )
            distill_total = distill_losses.get(
                "total", torch.tensor(0.0, device=student_depth.device)
            )
            total = total + (self.distillation_weight * distill_total)
            losses.update(distill_losses)

        # GT supervision.
        if gt_depth is not None:
            if self.target_mode == "relative":
                gt_rel = self._normalize_relative_depth(gt_depth, mask)
                pred_rel = student_relative

                if self.relative_loss_type == "ssi":
                    aligned_pred = self.ssi.align_scale_shift(
                        pred_rel, gt_rel, mask=mask
                    )
                    ssi_loss = self.ssi(
                        pred_rel, gt_rel, mask=mask, aligned_pred=aligned_pred
                    )
                else:
                    ssi_loss = self._masked_l1(pred_rel, gt_rel, mask=mask)
                    aligned_pred = pred_rel
                total = total + (ssi_weight * ssi_loss)
                losses["ssi"] = ssi_loss.detach()

                if gradient_weight > 0:
                    gradient_loss = self.gradient(aligned_pred, gt_rel, mask=mask)
                    total = total + (gradient_weight * gradient_loss)
                    losses["gradient"] = gradient_loss.detach()

                if boundary_weight > 0:
                    boundary_mask = self._boundary_mask(
                        gt_rel, mask=mask, log_space=False
                    )
                    boundary_l1 = self._masked_l1(
                        aligned_pred, gt_rel, mask=boundary_mask
                    )
                    total = total + (boundary_weight * boundary_l1)
                    losses["boundary_l1"] = boundary_l1.detach()

                if self._temporal_loss is not None and temporal_weight > 0:
                    try:
                        tgm_loss = self._temporal_loss(aligned_pred, gt_rel, mask)
                        total = total + (temporal_weight * tgm_loss)
                        losses["tgm"] = tgm_loss.detach()
                    except NotImplementedError:
                        pass
            else:
                aligned_pred = self.ssi.align_scale_shift(
                    student_depth, gt_depth, mask=mask
                )
                ssi_loss = self.ssi(
                    student_depth, gt_depth, mask=mask, aligned_pred=aligned_pred
                )
                total = total + (ssi_weight * ssi_loss)
                losses["ssi"] = ssi_loss.detach()

                if gradient_weight > 0:
                    gradient_loss = self.gradient(aligned_pred, gt_depth, mask=mask)
                    total = total + (gradient_weight * gradient_loss)
                    losses["gradient"] = gradient_loss.detach()

                if boundary_weight > 0:
                    boundary_mask = self._boundary_mask(
                        gt_depth, mask=mask, log_space=True
                    )
                    boundary_log_l1 = self._masked_l1(
                        torch.log(student_depth.clamp(min=self.eps)),
                        torch.log(gt_depth.clamp(min=self.eps)),
                        mask=boundary_mask,
                    )
                    total = total + (boundary_weight * boundary_log_l1)
                    losses["boundary_log_l1"] = boundary_log_l1.detach()

                if metric_l1_weight > 0:
                    metric_l1 = self._masked_l1(student_depth, gt_depth, mask=mask)
                    total = total + (metric_l1_weight * metric_l1)
                    losses["metric_l1"] = metric_l1.detach()

                if metric_log_l1_weight > 0:
                    metric_log_l1 = self._masked_l1(
                        torch.log(student_depth.clamp(min=self.eps)),
                        torch.log(gt_depth.clamp(min=self.eps)),
                        mask=mask,
                    )
                    total = total + (metric_log_l1_weight * metric_log_l1)
                    losses["metric_log_l1"] = metric_log_l1.detach()

                if range_prior_weight > 0:
                    range_prior = self._range_prior(student_depth, mask=mask)
                    total = total + (range_prior_weight * range_prior)
                    losses["range_prior"] = range_prior.detach()

                if self._temporal_loss is not None and temporal_weight > 0:
                    try:
                        tgm_loss = self._temporal_loss(aligned_pred, gt_depth, mask)
                        total = total + (temporal_weight * tgm_loss)
                        losses["tgm"] = tgm_loss.detach()
                    except NotImplementedError:
                        pass

            if self.edge_loss_weight > 0 and "edge_logits" in student_outputs:
                edge_target = self._boundary_mask(
                    gt_depth
                    if self.target_mode == "metric"
                    else self._normalize_relative_depth(gt_depth, mask),
                    mask=mask,
                    log_space=self.target_mode == "metric",
                ).float()
                edge_logits = student_outputs["edge_logits"]
                edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_target)
                total = total + (self.edge_loss_weight * edge_loss)
                losses["edge_bce"] = edge_loss.detach()

        losses["total"] = total
        return losses

    @property
    def active_teacher_names(self) -> tuple[str, ...]:
        if self.distillation is None:
            return ()
        return self.distillation.configured_teacher_names

    @property
    def wants_auxiliary_clean_pass(self) -> bool:
        return (
            self.feature_alignment_weight > 0
            or self.clean_depth_consistency_weight > 0
            or self.recon_l1_weight > 0
            or self.recon_ssim_weight > 0
        )

    @property
    def wants_feature_summary(self) -> bool:
        return self.feature_alignment_weight > 0

    def auxiliary_consistency_losses(
        self,
        student_outputs: Dict[str, torch.Tensor],
        clean_outputs: Dict[str, torch.Tensor],
        *,
        mask: Optional[torch.Tensor] = None,
        clean_frames: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        device = student_outputs["depth"].device
        total = self._zero_loss(device)

        if self.feature_alignment_weight > 0:
            student_summary = student_outputs.get("feature_summary")
            clean_summary = clean_outputs.get("feature_summary")
            if student_summary is not None and clean_summary is not None:
                feature_loss = (student_summary - clean_summary.detach()).abs().mean()
                total = total + self.feature_alignment_weight * feature_loss
                losses["aux/feature_alignment"] = feature_loss.detach()

        if self.clean_depth_consistency_weight > 0:
            student_depth = student_outputs["depth"]
            clean_depth = clean_outputs["depth"].detach()
            if self.clean_depth_consistency_mode == "si":
                aligned_student = self.ssi.align_scale_shift(
                    student_depth, clean_depth, mask=mask
                )
                depth_consistency = self.ssi(
                    student_depth, clean_depth, mask=mask, aligned_pred=aligned_student
                )
            elif self.clean_depth_consistency_mode == "l1":
                depth_consistency = self._masked_l1(
                    student_depth, clean_depth, mask=mask
                )
            else:
                depth_consistency = self._masked_l1(
                    torch.log(student_depth.clamp(min=self.eps)),
                    torch.log(clean_depth.clamp(min=self.eps)),
                    mask=mask,
                )
            total = total + self.clean_depth_consistency_weight * depth_consistency
            losses["aux/clean_depth_consistency"] = depth_consistency.detach()

        if self.recon_l1_weight > 0 and clean_frames is not None:
            reconstruction = student_outputs.get("reconstruction")
            if reconstruction is not None:
                recon_l1 = self._masked_l1(reconstruction, clean_frames.detach())
                total = total + self.recon_l1_weight * recon_l1
                losses["aux/recon_l1"] = recon_l1.detach()

                if self.recon_ssim_weight > 0:
                    recon_ssim = 1.0 - self._simple_ssim(
                        reconstruction, clean_frames.detach()
                    )
                    total = total + self.recon_ssim_weight * recon_ssim
                    losses["aux/recon_ssim"] = recon_ssim.detach()

        losses["aux/total"] = total
        return losses

    def _zero_loss(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(0.0, device=device)

    def _simple_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1 = 0.01**2
        c2 = 0.03**2
        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(
            -1, pred.shape[1], pred.shape[-2], pred.shape[-1]
        )
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(
            -1, target.shape[1], target.shape[-2], target.shape[-1]
        )
        mu_x = F.avg_pool2d(pred_2d, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(target_2d, kernel_size=3, stride=1, padding=1)
        sigma_x = (
            F.avg_pool2d(pred_2d * pred_2d, kernel_size=3, stride=1, padding=1)
            - mu_x * mu_x
        )
        sigma_y = (
            F.avg_pool2d(target_2d * target_2d, kernel_size=3, stride=1, padding=1)
            - mu_y * mu_y
        )
        sigma_xy = (
            F.avg_pool2d(pred_2d * target_2d, kernel_size=3, stride=1, padding=1)
            - mu_x * mu_y
        )
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ).clamp_min(self.eps)
        return ssim_map.mean()
