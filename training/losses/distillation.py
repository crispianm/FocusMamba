"""
Multi-teacher distillation losses for metric depth training.

Supports:
- ``legacy``: existing SI-log / MTKD behavior, but with strict teacher filtering.
- ``calibrated_aggregate``: GT-calibrated multi-teacher aggregation.
- ``heterogeneous_mtkd`` / ``anchor_vda``: GT-calibrated aggregation plus
  teacher-specific auxiliary losses for heterogeneous teachers.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .gradient import GradientSmoothnessLoss
from .scale_invariant import ScaleInvariantLogLoss


class DistillationLoss(nn.Module):
    """Multi-teacher distillation loss with optional heterogeneous supervision."""

    _VALID_STRATEGIES = {
        "legacy",
        "calibrated_aggregate",
        "heterogeneous_mtkd",
        "anchor_vda",
    }

    def __init__(
        self,
        distillation_cfg: Dict,
        target_mode: str = "metric",
        relative_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.teacher_configs: List[Dict] = list(distillation_cfg.get("teachers", []))
        self.teacher_names = [
            cfg["name"] for cfg in self.teacher_configs if "name" in cfg
        ]
        self.confidence_weighted = bool(
            distillation_cfg.get("confidence_weighted", True)
        )
        self.lambda_si = float(distillation_cfg.get("lambda_si", 0.5))
        self.eps = 1e-6
        self.si_log = ScaleInvariantLogLoss(lambda_si=self.lambda_si, eps=self.eps)

        self.target_mode = str(target_mode).lower()
        if self.target_mode not in ("metric", "relative"):
            raise ValueError(f"Unknown distillation target_mode={target_mode!r}")

        relative_cfg = relative_cfg or {}
        self.relative_log_space = bool(relative_cfg.get("log_space", True))
        self.relative_q_low = float(relative_cfg.get("q_low", 0.02))
        self.relative_q_high = float(relative_cfg.get("q_high", 0.98))
        self.relative_clamp = bool(relative_cfg.get("clamp", True))

        self.strategy = str(distillation_cfg.get("strategy", "legacy")).lower()
        if self.strategy not in self._VALID_STRATEGIES:
            valid = ", ".join(sorted(self._VALID_STRATEGIES))
            raise ValueError(
                f"Unknown distillation.strategy={self.strategy!r}. Valid: {valid}"
            )

        self.max_depth = float(distillation_cfg.get("max_depth", 80.0))
        calibration_cfg = distillation_cfg.get("calibration", {}) or {}
        self.calibration_enabled = bool(
            calibration_cfg.get("enabled", self.strategy != "legacy")
        )
        self.calibration_min_depth = float(calibration_cfg.get("min_depth", 0.1))
        self.calibration_min_valid_pixels = int(
            calibration_cfg.get("min_valid_pixels", 64)
        )
        self.calibration_use_gt = bool(calibration_cfg.get("use_gt", True))

        mtkd_cfg = distillation_cfg.get("mtkd", {}) or {}
        self.mtkd_enabled = bool(mtkd_cfg.get("enabled", False))
        self.mtkd_alpha = float(mtkd_cfg.get("alpha", 1.0))
        self.mtkd_alpha = max(0.0, min(1.0, self.mtkd_alpha))
        self.mtkd_wavelet_weight = float(mtkd_cfg.get("wavelet_weight", 0.0))
        self.mtkd_wavelet_levels = int(mtkd_cfg.get("wavelet_levels", 1))
        self.mtkd_reliability_beta = float(mtkd_cfg.get("reliability_beta", 6.0))
        self.mtkd_temporal_weight = float(mtkd_cfg.get("temporal_weight", 0.0))

        aggregate_cfg = distillation_cfg.get("aggregate", {}) or {}
        default_agg_weight = 1.0 if self.strategy == "calibrated_aggregate" else 0.5
        self.aggregate_weight = float(aggregate_cfg.get("weight", default_agg_weight))
        self.aggregate_beta = float(
            aggregate_cfg.get("beta", self.mtkd_reliability_beta)
        )

        priors_cfg = distillation_cfg.get("teacher_priors", {}) or {}
        self.teacher_priors = {
            cfg["name"]: float(priors_cfg.get(cfg["name"], cfg.get("weight", 1.0)))
            for cfg in self.teacher_configs
            if "name" in cfg
        }

        aux_cfg = distillation_cfg.get("auxiliary", {}) or {}
        self.vda_temporal_weight = float(aux_cfg.get("vda_temporal_weight", 0.5))
        self.structure_weight = float(aux_cfg.get("structure_weight", 0.5))
        self.gradient_scales = int(aux_cfg.get("gradient_scales", 1))
        self.aux_teacher_weights = {
            "video_depth_anything": float(aux_cfg.get("vda_weight", 0.3)),
            "depth_anything_v3": float(aux_cfg.get("da3_weight", 0.1)),
            "depth_pro": float(aux_cfg.get("depth_pro_weight", 0.1)),
        }
        self.default_aux_weight = float(aux_cfg.get("default_weight", 0.1))

        self.gradient_metric = GradientSmoothnessLoss(
            mode="matching",
            scales=self.gradient_scales,
            log_space=True,
        )
        self.gradient_relative = GradientSmoothnessLoss(
            mode="matching",
            scales=self.gradient_scales,
            log_space=False,
        )

    @property
    def configured_teacher_names(self) -> tuple[str, ...]:
        return tuple(self.teacher_names)

    def forward(
        self,
        student_depth: torch.Tensor,
        teacher_depths: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        student_weight_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss for the configured teacher subset only."""
        filtered_teachers = self._filter_teacher_depths(teacher_depths)
        if not filtered_teachers:
            return {"total": self._zero_loss(student_depth.device)}

        if self.strategy == "legacy":
            return self._forward_legacy(
                student_depth,
                filtered_teachers,
                mask=mask,
                student_weight_map=student_weight_map,
            )
        return self._forward_heterogeneous(
            student_depth,
            filtered_teachers,
            gt_depth=gt_depth,
            mask=mask,
            student_weight_map=student_weight_map,
        )

    def _forward_legacy(
        self,
        student_depth: torch.Tensor,
        teacher_depths: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        student_weight_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        weighted_sum = torch.tensor(0.0, device=student_depth.device)
        total_weight = 0.0

        mode = self.target_mode
        if mode == "relative":
            student_target = self._normalize_relative_depth(student_depth, mask)
            teacher_targets = {
                name: self._normalize_relative_depth(depth, mask)
                for name, depth in teacher_depths.items()
            }
        else:
            student_target = student_depth
            teacher_targets = teacher_depths

        confidence = None
        if self.confidence_weighted and len(teacher_targets) > 1:
            confidence = self._compute_confidence(
                teacher_targets,
                log_space=(mode == "metric"),
            )
            losses["confidence_mean"] = confidence.mean().detach()
        confidence = self._combine_weight_maps(confidence, student_weight_map)
        if student_weight_map is not None:
            losses["student_weight_mean"] = student_weight_map.mean().detach()

        for cfg in self.teacher_configs:
            name = cfg["name"]
            if name not in teacher_targets:
                continue
            weight = float(cfg.get("weight", 1.0))
            loss_type = cfg.get("loss", "si_log")
            teacher_d = teacher_targets[name]

            if mode == "relative":
                main_loss = self._relative_l1_weighted(
                    student_target,
                    teacher_d,
                    mask=mask,
                    weight_map=confidence,
                )
            else:
                main_loss = self._si_log_weighted(
                    student_target,
                    teacher_d,
                    mask=mask,
                    weight_map=confidence,
                )
            losses[f"distill/{name}_si_log"] = main_loss.detach()
            teacher_loss = main_loss

            if loss_type == "si_log_plus_temporal" and student_target.shape[2] > 1:
                temporal_loss = self._temporal_consistency(
                    student_target,
                    teacher_d,
                    mask=mask,
                    log_space=(mode == "metric"),
                )
                losses[f"distill/{name}_temporal"] = temporal_loss.detach()
                teacher_loss = teacher_loss + 0.5 * temporal_loss

            weighted_sum = weighted_sum + weight * teacher_loss
            total_weight += weight

        base_total = None
        if total_weight > 0:
            base_total = weighted_sum / total_weight

        mtkd_total = None
        if self.mtkd_enabled and len(teacher_targets) > 0:
            agg_depth, teacher_mix = self._aggregate_teachers(
                teacher_targets, mode=mode
            )

            if mode == "relative":
                agg_main = self._relative_l1_weighted(
                    student_target,
                    agg_depth,
                    mask=mask,
                    weight_map=confidence,
                )
            else:
                agg_main = self._si_log_weighted(
                    student_target,
                    agg_depth,
                    mask=mask,
                    weight_map=confidence,
                )
            losses["distill/mtkd_agg_si_log"] = agg_main.detach()

            dwt_loss = torch.tensor(0.0, device=student_depth.device)
            if self.mtkd_wavelet_weight > 0:
                dwt_loss = self._haar_dwt_l1(
                    student_target,
                    agg_depth,
                    mask=mask,
                    levels=max(1, self.mtkd_wavelet_levels),
                    log_space=(mode == "metric"),
                )
                losses["distill/mtkd_wavelet"] = dwt_loss.detach()

            t_loss = torch.tensor(0.0, device=student_depth.device)
            if self.mtkd_temporal_weight > 0 and student_target.shape[2] > 1:
                t_loss = self._temporal_consistency(
                    student_target,
                    agg_depth,
                    mask=mask,
                    log_space=(mode == "metric"),
                )
                losses["distill/mtkd_temporal"] = t_loss.detach()

            mtkd_total = (
                agg_main
                + self.mtkd_wavelet_weight * dwt_loss
                + self.mtkd_temporal_weight * t_loss
            )
            losses["distill/mtkd_total"] = mtkd_total.detach()
            for name, w in teacher_mix.items():
                losses[f"distill/mtkd_mix_{name}"] = w.detach()

        if base_total is not None and mtkd_total is not None:
            losses["total"] = (
                1.0 - self.mtkd_alpha
            ) * base_total + self.mtkd_alpha * mtkd_total
        elif mtkd_total is not None:
            losses["total"] = mtkd_total
        elif base_total is not None:
            losses["total"] = base_total
        else:
            losses["total"] = self._zero_loss(student_depth.device)
        return losses

    def _forward_heterogeneous(
        self,
        student_depth: torch.Tensor,
        teacher_depths: Dict[str, torch.Tensor],
        *,
        gt_depth: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        student_weight_map: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.target_mode != "metric":
            raise ValueError(
                f"Heterogeneous distillation strategies require metric supervision, got {self.target_mode!r}"
            )

        prepared = self._prepare_teacher_targets(
            teacher_depths, gt_depth=gt_depth, mask=mask
        )
        if not prepared:
            return {"total": self._zero_loss(student_depth.device)}

        losses: Dict[str, torch.Tensor] = {}
        components: list[tuple[str, float, torch.Tensor]] = []

        agg_depth, agg_valid, teacher_mix = self._compute_calibrated_aggregate(prepared)
        if agg_valid.any():
            agg_loss = self._si_log_weighted(
                student_depth,
                agg_depth,
                mask=agg_valid,
                weight_map=student_weight_map,
            )
            losses["distill/agg_si_log"] = agg_loss.detach()
            losses["distill/agg_valid_fraction"] = agg_valid.float().mean().detach()
            components.append(("agg", self.aggregate_weight, agg_loss))
        if student_weight_map is not None:
            losses["distill/student_weight_mean"] = student_weight_map.mean().detach()
        for name in self.teacher_names:
            mix = teacher_mix.get(
                name,
                torch.tensor(
                    0.0, device=student_depth.device, dtype=student_depth.dtype
                ),
            )
            losses[f"distill/mtkd_mix_{name}"] = mix.detach()

        if self.strategy in {"heterogeneous_mtkd", "anchor_vda"}:
            for name, info in prepared.items():
                aux_loss, aux_metrics = self._compute_teacher_aux_loss(
                    name,
                    student_depth,
                    info["calibrated_depth"],
                    info["valid_mask"],
                    weight_map=student_weight_map,
                )
                if aux_loss is None:
                    continue
                aux_weight = self._get_aux_weight(name)
                components.append((name, aux_weight, aux_loss))
                losses[f"distill/{name}_aux"] = aux_loss.detach()
                for metric_name, metric_value in aux_metrics.items():
                    losses[f"distill/{name}_{metric_name}"] = metric_value.detach()

        for name, info in prepared.items():
            losses[f"distill/{name}_valid_fraction"] = (
                info["valid_mask"].float().mean().detach()
            )
            losses[f"distill/{name}_shift"] = info["shift"].mean().detach()

        total = self._renormalize_components(components, device=student_depth.device)
        losses["total"] = total
        return losses

    def _filter_teacher_depths(
        self,
        teacher_depths: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            name: teacher_depths[name]
            for name in self.teacher_names
            if name in teacher_depths
        }

    def _prepare_teacher_targets(
        self,
        teacher_depths: Dict[str, torch.Tensor],
        *,
        gt_depth: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        gt_valid = None
        if gt_depth is not None:
            gt_valid = gt_depth > 0
        if mask is not None:
            gt_valid = mask.bool() if gt_valid is None else (gt_valid & mask.bool())

        prepared: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, depth in teacher_depths.items():
            teacher_valid = self._teacher_valid_mask(name, depth, gt_valid)
            calibrated_depth, shift, calibrated_valid = self._calibrate_teacher_depth(
                depth=depth,
                gt_depth=gt_depth,
                valid_mask=teacher_valid,
            )
            if not calibrated_valid.any():
                continue
            prepared[name] = {
                "raw_depth": depth,
                "calibrated_depth": calibrated_depth,
                "valid_mask": calibrated_valid,
                "shift": shift,
            }
        return prepared

    def _teacher_valid_mask(
        self,
        name: str,
        depth: torch.Tensor,
        gt_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        valid = (
            torch.isfinite(depth)
            & (depth >= self.calibration_min_depth)
            & (depth <= self.max_depth)
        )
        if name == "depth_anything_v3":
            valid = valid & (depth > 0)
        if gt_valid is not None:
            valid = valid & gt_valid.bool()
        return valid

    def _calibrate_teacher_depth(
        self,
        *,
        depth: torch.Tensor,
        gt_depth: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            not self.calibration_enabled
            or not self.calibration_use_gt
            or gt_depth is None
        ):
            return depth, torch.zeros_like(depth[:, :, :1, :1, :1]), valid_mask

        gt_log = torch.log(gt_depth.clamp(min=self.eps))
        teacher_log = torch.log(depth.clamp(min=self.eps))
        batch_size = depth.shape[0]
        shift = torch.zeros(
            (batch_size, 1, 1, 1, 1),
            device=depth.device,
            dtype=depth.dtype,
        )

        for batch_idx in range(batch_size):
            valid = valid_mask[batch_idx].bool()
            if valid.sum() < self.calibration_min_valid_pixels:
                continue
            delta = torch.median(
                gt_log[batch_idx][valid] - teacher_log[batch_idx][valid]
            )
            shift[batch_idx] = delta

        calibrated = torch.exp(teacher_log + shift)
        calibrated = calibrated.clamp(
            min=self.calibration_min_depth, max=self.max_depth
        )
        calibrated_valid = valid_mask & torch.isfinite(calibrated)
        calibrated_valid = (
            calibrated_valid
            & (calibrated >= self.calibration_min_depth)
            & (calibrated <= self.max_depth)
        )
        return calibrated, shift, calibrated_valid

    def _compute_calibrated_aggregate(
        self,
        prepared: Dict[str, Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        names = list(prepared.keys())
        if not names:
            raise ValueError("Cannot aggregate zero teachers")

        log_payload = []
        valid_payload = []
        prior_payload = []
        for name in names:
            depth = prepared[name]["calibrated_depth"]
            valid = prepared[name]["valid_mask"].bool()
            log_depth = torch.log(depth.clamp(min=self.eps))
            log_payload.append(
                torch.where(valid, log_depth, torch.full_like(log_depth, float("nan")))
            )
            valid_payload.append(valid)
            prior_payload.append(self.teacher_priors.get(name, 1.0))

        stacked_logs = torch.stack(log_payload, dim=0)
        stacked_valid = torch.stack(valid_payload, dim=0)
        center = torch.nanmedian(stacked_logs, dim=0).values
        center = torch.where(torch.isfinite(center), center, torch.zeros_like(center))

        priors = torch.tensor(
            prior_payload,
            device=stacked_logs.device,
            dtype=stacked_logs.dtype,
        ).view(-1, 1, 1, 1, 1, 1)
        reliability = priors * torch.exp(
            -self.aggregate_beta * (stacked_logs - center).abs()
        )
        reliability = reliability * stacked_valid.to(dtype=stacked_logs.dtype)
        reliability_sum = reliability.sum(dim=0, keepdim=True)
        agg_valid = reliability_sum.squeeze(0) > 0
        weights = reliability / reliability_sum.clamp(min=self.eps)
        agg_log = torch.where(
            agg_valid,
            torch.nansum(weights * stacked_logs, dim=0),
            torch.zeros_like(center),
        )
        agg_depth = torch.exp(agg_log)

        teacher_mix = {}
        for idx, name in enumerate(names):
            if agg_valid.any():
                teacher_mix[name] = weights[idx][agg_valid].mean()
            else:
                teacher_mix[name] = torch.tensor(
                    0.0, device=agg_depth.device, dtype=agg_depth.dtype
                )
        return agg_depth, agg_valid, teacher_mix

    def _compute_teacher_aux_loss(
        self,
        name: str,
        student_depth: torch.Tensor,
        teacher_depth: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        weight_map: torch.Tensor | None = None,
    ) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not valid_mask.any():
            return None, {}

        mode = self._resolve_teacher_mode(name)
        metrics: Dict[str, torch.Tensor] = {}

        if mode == "metric_temporal":
            si = self._si_log_weighted(
                student_depth,
                teacher_depth,
                mask=valid_mask,
                weight_map=weight_map,
            )
            temporal = self._temporal_consistency(
                student_depth,
                teacher_depth,
                mask=valid_mask,
                log_space=True,
            )
            metrics["si_log"] = si.detach()
            metrics["temporal"] = temporal.detach()
            combined = si + self.vda_temporal_weight * temporal
            return combined * self._mean_weight(weight_map, valid_mask), metrics

        if mode == "metric_structure":
            si = self._si_log_weighted(
                student_depth,
                teacher_depth,
                mask=valid_mask,
                weight_map=weight_map,
            )
            grad = self.gradient_metric(
                student_depth, teacher_depth, mask=valid_mask.float()
            )
            metrics["si_log"] = si.detach()
            metrics["gradient"] = grad.detach()
            combined = si + self.structure_weight * grad
            return combined * self._mean_weight(weight_map, valid_mask), metrics

        if mode == "metric":
            si = self._si_log_weighted(
                student_depth,
                teacher_depth,
                mask=valid_mask,
                weight_map=weight_map,
            )
            metrics["si_log"] = si.detach()
            return si * self._mean_weight(weight_map, valid_mask), metrics

        if mode in {"relative_structure", "structure_only"}:
            student_rel = self._normalize_relative_depth(student_depth, valid_mask)
            teacher_rel = self._normalize_relative_depth(teacher_depth, valid_mask)
            rel_l1 = self._relative_l1_weighted(
                student_rel,
                teacher_rel,
                mask=valid_mask,
                weight_map=weight_map,
            )
            grad = self.gradient_relative(
                student_rel, teacher_rel, mask=valid_mask.float()
            )
            metrics["rel_l1"] = rel_l1.detach()
            metrics["gradient"] = grad.detach()
            combined = rel_l1 + self.structure_weight * grad
            return combined * self._mean_weight(weight_map, valid_mask), metrics

        raise ValueError(f"Unsupported distill_mode={mode!r} for teacher {name!r}")

    def _resolve_teacher_mode(self, name: str) -> str:
        cfg = next((cfg for cfg in self.teacher_configs if cfg.get("name") == name), {})
        explicit_mode = cfg.get("distill_mode")
        if explicit_mode is not None:
            return str(explicit_mode).lower()
        if self.strategy == "anchor_vda" and name != "video_depth_anything":
            return "structure_only"
        if name == "video_depth_anything":
            return "metric_temporal"
        if name == "depth_anything_v3":
            return "relative_structure"
        if name == "depth_pro":
            return "metric_structure"
        return "metric"

    def _get_aux_weight(self, name: str) -> float:
        cfg = next((cfg for cfg in self.teacher_configs if cfg.get("name") == name), {})
        if "aux_weight" in cfg:
            return float(cfg["aux_weight"])
        return float(self.aux_teacher_weights.get(name, self.default_aux_weight))

    def _renormalize_components(
        self,
        components: list[tuple[str, float, torch.Tensor]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        active_components = [
            (name, weight, loss)
            for name, weight, loss in components
            if weight > 0 and torch.isfinite(loss)
        ]
        if not active_components:
            return self._zero_loss(device)

        weight_sum = sum(weight for _, weight, _ in active_components)
        total = torch.tensor(0.0, device=device)
        for _, weight, loss in active_components:
            total = total + (weight / weight_sum) * loss
        return total

    @staticmethod
    def _combine_weight_maps(
        base: torch.Tensor | None,
        extra: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if base is None:
            return extra
        if extra is None:
            return base
        return base * extra

    @staticmethod
    def _mean_weight(
        weight_map: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if weight_map is None:
            device = mask.device if mask is not None else "cpu"
            return torch.tensor(1.0, device=device, dtype=torch.float32)
        valid = torch.isfinite(weight_map)
        if mask is not None:
            valid = valid & mask.bool()
        if valid.any():
            return weight_map[valid].mean()
        return torch.tensor(1.0, device=weight_map.device, dtype=weight_map.dtype)

    def _normalize_relative_depth(
        self,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.relative_log_space:
            base = torch.log(depth.clamp(min=self.eps))
        else:
            base = depth

        valid = torch.isfinite(base)
        if mask is not None:
            valid = valid & mask.bool()

        norm = torch.zeros_like(base)
        batch_size = base.shape[0]
        for batch_idx in range(batch_size):
            valid_batch = valid[batch_idx]
            if valid_batch.sum() < 8:
                continue
            vals = base[batch_idx][valid_batch].detach()
            q_low = torch.quantile(vals, self.relative_q_low)
            q_high = torch.quantile(vals, self.relative_q_high)
            denom = (q_high - q_low).clamp(min=self.eps)
            normalized = (base[batch_idx] - q_low) / denom
            if self.relative_clamp:
                normalized = normalized.clamp(0.0, 1.0)
            norm[batch_idx] = torch.where(
                valid_batch, normalized, torch.zeros_like(normalized)
            )
        return norm

    def _si_log_weighted(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_diff = torch.log(pred.clamp(min=self.eps)) - torch.log(
            gt.clamp(min=self.eps)
        )

        valid = torch.isfinite(log_diff)
        if mask is not None:
            valid = valid & mask.bool()

        if weight_map is None:
            weights = valid.to(dtype=log_diff.dtype)
        else:
            weights = weight_map.to(dtype=log_diff.dtype) * valid.to(
                dtype=log_diff.dtype
            )

        weight_sum = weights.sum()
        if weight_sum <= 0:
            return self._zero_loss(pred.device)

        mean_d2 = (weights * (log_diff**2)).sum() / weight_sum
        mean_d = (weights * log_diff).sum() / weight_sum
        return mean_d2 - self.lambda_si * (mean_d**2)

    @staticmethod
    def _relative_l1_weighted(
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = (pred - gt).abs()
        valid = torch.isfinite(diff)
        if mask is not None:
            valid = valid & mask.bool()

        if weight_map is None:
            weights = valid.to(dtype=diff.dtype)
        else:
            weights = weight_map.to(dtype=diff.dtype) * valid.to(dtype=diff.dtype)

        weight_sum = weights.sum()
        if weight_sum <= 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (weights * diff).sum() / weight_sum

    @staticmethod
    def _temporal_consistency(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        log_space: bool = True,
    ) -> torch.Tensor:
        if pred.shape[2] <= 1:
            return torch.tensor(0.0, device=pred.device)

        if log_space:
            pred_space = torch.log(pred.clamp(min=1e-6))
            target_space = torch.log(target.clamp(min=1e-6))
        else:
            pred_space = pred
            target_space = target

        td_pred = pred_space[:, :, 1:] - pred_space[:, :, :-1]
        td_target = target_space[:, :, 1:] - target_space[:, :, :-1]
        td_diff = (td_pred - td_target).abs()

        if mask is not None:
            valid = mask.bool()
            valid_t = valid[:, :, 1:] & valid[:, :, :-1]
            if valid_t.any():
                return td_diff[valid_t].mean()
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return td_diff.mean()

    @staticmethod
    def _compute_confidence(
        teacher_depths: Dict[str, torch.Tensor],
        log_space: bool = True,
    ) -> torch.Tensor:
        depths = list(teacher_depths.values())
        if len(depths) < 2:
            return torch.ones_like(depths[0])

        if log_space:
            stacked = torch.stack(
                [torch.log(depth.clamp(min=1e-6)) for depth in depths], dim=0
            )
        else:
            stacked = torch.stack(depths, dim=0)
        variance = stacked.var(dim=0)
        return torch.exp(-5.0 * variance)

    def _aggregate_teachers(
        self,
        teacher_depths: Dict[str, torch.Tensor],
        mode: str = "metric",
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        names = []
        static_weights = []
        payload = []
        for cfg in self.teacher_configs:
            name = cfg["name"]
            if name not in teacher_depths:
                continue
            names.append(name)
            static_weights.append(float(cfg.get("weight", 1.0)))
            if mode == "metric":
                payload.append(torch.log(teacher_depths[name].clamp(min=self.eps)))
            else:
                payload.append(teacher_depths[name])

        if not payload:
            any_depth = next(iter(teacher_depths.values()))
            return any_depth, {}

        stacked = torch.stack(payload, dim=0)
        static_w = torch.tensor(
            static_weights, device=stacked.device, dtype=stacked.dtype
        )
        static_w = static_w / static_w.sum().clamp(min=self.eps)
        static_w = static_w.view(-1, 1, 1, 1, 1, 1)

        center = stacked.mean(dim=0, keepdim=True)
        reliability = torch.exp(-self.mtkd_reliability_beta * (stacked - center).abs())
        weight = static_w * reliability
        weight = weight / weight.sum(dim=0, keepdim=True).clamp(min=self.eps)

        agg_val = (weight * stacked).sum(dim=0)
        if mode == "metric":
            agg_depth = torch.exp(agg_val)
        else:
            agg_depth = agg_val

        teacher_mix = {name: weight[idx].mean() for idx, name in enumerate(names)}
        return agg_depth, teacher_mix

    def _haar_dwt_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        levels: int = 1,
        log_space: bool = True,
    ) -> torch.Tensor:
        if log_space:
            pred_space = torch.log(pred.clamp(min=self.eps))
            target_space = torch.log(target.clamp(min=self.eps))
        else:
            pred_space = pred
            target_space = target

        if mask is not None:
            valid = mask.bool()
            pred_space = torch.where(valid, pred_space, target_space)

        x = pred_space.flatten(0, 2)
        y = target_space.flatten(0, 2)

        detail_terms = []
        ll_x, ll_y = x, y
        for _ in range(max(1, levels)):
            if ll_x.shape[-2] < 2 or ll_x.shape[-1] < 2:
                break
            if (ll_x.shape[-2] % 2) != 0 or (ll_x.shape[-1] % 2) != 0:
                ll_x = torch.nn.functional.pad(
                    ll_x,
                    (0, ll_x.shape[-1] % 2, 0, ll_x.shape[-2] % 2),
                    mode="replicate",
                )
                ll_y = torch.nn.functional.pad(
                    ll_y,
                    (0, ll_y.shape[-1] % 2, 0, ll_y.shape[-2] % 2),
                    mode="replicate",
                )

            xa = ll_x[..., 0::2, 0::2]
            xb = ll_x[..., 0::2, 1::2]
            xc = ll_x[..., 1::2, 0::2]
            xd = ll_x[..., 1::2, 1::2]

            ya = ll_y[..., 0::2, 0::2]
            yb = ll_y[..., 0::2, 1::2]
            yc = ll_y[..., 1::2, 0::2]
            yd = ll_y[..., 1::2, 1::2]

            ll_x = 0.5 * (xa + xb + xc + xd)
            ll_y = 0.5 * (ya + yb + yc + yd)
            lh_x = 0.5 * (xa - xb + xc - xd)
            lh_y = 0.5 * (ya - yb + yc - yd)
            hl_x = 0.5 * (xa + xb - xc - xd)
            hl_y = 0.5 * (ya + yb - yc - yd)
            hh_x = 0.5 * (xa - xb - xc + xd)
            hh_y = 0.5 * (ya - yb - yc + yd)

            detail_terms.append((lh_x - lh_y).abs().mean())
            detail_terms.append((hl_x - hl_y).abs().mean())
            detail_terms.append((hh_x - hh_y).abs().mean())

        approx_term = (ll_x - ll_y).abs().mean()
        all_terms = [approx_term] + detail_terms
        return torch.stack(all_terms).mean()

    @staticmethod
    def _zero_loss(device: torch.device) -> torch.Tensor:
        return torch.tensor(0.0, device=device, requires_grad=True)
