"""
Multi-Teacher Distillation Loss
================================

Supports two supervision modes:
- metric:   SI-log distillation in metric depth space.
- relative: robustly normalised relative-depth distillation.
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
        target_mode: 'metric' or 'relative'.
        relative_cfg: Optional config for relative-depth normalization.
    """

    def __init__(
        self,
        teacher_configs: List[Dict],
        confidence_weighted: bool = True,
        lambda_si: float = 0.5,
        mtkd_cfg: Optional[Dict] = None,
        target_mode: str = "metric",
        relative_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.teacher_configs = teacher_configs
        self.confidence_weighted = confidence_weighted
        self.lambda_si = float(lambda_si)
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

        # Optional MTKD-style extensions:
        mtkd_cfg = mtkd_cfg or {}
        self.mtkd_enabled = bool(mtkd_cfg.get("enabled", False))
        self.mtkd_alpha = float(mtkd_cfg.get("alpha", 1.0))
        self.mtkd_alpha = max(0.0, min(1.0, self.mtkd_alpha))
        self.mtkd_wavelet_weight = float(mtkd_cfg.get("wavelet_weight", 0.0))
        self.mtkd_wavelet_levels = int(mtkd_cfg.get("wavelet_levels", 1))
        self.mtkd_reliability_beta = float(mtkd_cfg.get("reliability_beta", 6.0))
        self.mtkd_temporal_weight = float(mtkd_cfg.get("temporal_weight", 0.0))

    def forward(
        self,
        student_depth: torch.Tensor,
        teacher_depths: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-teacher distillation loss."""
        losses = {}
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

        # Compute per-teacher losses
        for cfg in self.teacher_configs:
            name = cfg["name"]
            weight = cfg["weight"]
            loss_type = cfg.get("loss", "si_log")

            if name not in teacher_targets:
                continue

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

            # Add temporal consistency for video teachers.
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

        # MTKD-style aggregate-teacher supervision.
        mtkd_total = None
        if self.mtkd_enabled and len(teacher_targets) > 0:
            agg_depth, teacher_mix = self._aggregate_teachers(teacher_targets, mode=mode)

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

            mtkd_total = agg_main + self.mtkd_wavelet_weight * dwt_loss + self.mtkd_temporal_weight * t_loss
            losses["distill/mtkd_total"] = mtkd_total.detach()

            for name, w in teacher_mix.items():
                losses[f"distill/mtkd_mix_{name}"] = w.detach()

        if base_total is not None and mtkd_total is not None:
            losses["total"] = (1.0 - self.mtkd_alpha) * base_total + self.mtkd_alpha * mtkd_total
        elif mtkd_total is not None:
            losses["total"] = mtkd_total
        elif base_total is not None:
            losses["total"] = base_total
        else:
            losses["total"] = torch.tensor(0.0, device=student_depth.device, requires_grad=True)

        return losses

    def _normalize_relative_depth(
        self,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-clip robust normalization to relative depth in [0, 1]."""
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

    def _si_log_weighted(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_diff = torch.log(pred.clamp(min=self.eps)) - torch.log(gt.clamp(min=self.eps))

        valid = torch.isfinite(log_diff)
        if mask is not None:
            valid = valid & mask.bool()

        if weight_map is None:
            w = valid.to(dtype=log_diff.dtype)
        else:
            w = weight_map.to(dtype=log_diff.dtype) * valid.to(dtype=log_diff.dtype)

        w_sum = w.sum()
        if w_sum <= 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        mean_d2 = (w * (log_diff ** 2)).sum() / w_sum
        mean_d = (w * log_diff).sum() / w_sum
        return mean_d2 - self.lambda_si * (mean_d ** 2)

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
            w = valid.to(dtype=diff.dtype)
        else:
            w = weight_map.to(dtype=diff.dtype) * valid.to(dtype=diff.dtype)

        w_sum = w.sum()
        if w_sum <= 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (w * diff).sum() / w_sum

    @staticmethod
    def _temporal_consistency(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        log_space: bool = True,
    ) -> torch.Tensor:
        """Temporal consistency loss: L1 on temporal gradients."""
        if pred.shape[2] <= 1:
            return torch.tensor(0.0, device=pred.device)

        if log_space:
            p = torch.log(pred.clamp(min=1e-6))
            t = torch.log(target.clamp(min=1e-6))
        else:
            p = pred
            t = target

        td_pred = p[:, :, 1:] - p[:, :, :-1]
        td_target = t[:, :, 1:] - t[:, :, :-1]
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
        """Compute confidence map from teacher agreement."""
        depths = list(teacher_depths.values())
        if len(depths) < 2:
            return torch.ones_like(depths[0])

        if log_space:
            stacked = torch.stack([torch.log(d.clamp(min=1e-6)) for d in depths], dim=0)
        else:
            stacked = torch.stack(depths, dim=0)
        variance = stacked.var(dim=0)
        confidence = torch.exp(-5.0 * variance)
        return confidence

    def _aggregate_teachers(
        self,
        teacher_depths: Dict[str, torch.Tensor],
        mode: str = "metric",
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """MTKD-style dynamic aggregation in metric or relative space."""
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

        stacked = torch.stack(payload, dim=0)  # (N, B, 1, T, H, W)
        static_w = torch.tensor(static_weights, device=stacked.device, dtype=stacked.dtype)
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

        teacher_mix = {name: weight[i].mean() for i, name in enumerate(names)}
        return agg_depth, teacher_mix

    def _haar_dwt_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        levels: int = 1,
        log_space: bool = True,
    ) -> torch.Tensor:
        """Haar-DWT loss (log-depth for metric mode, linear for relative mode)."""
        if log_space:
            pred_space = torch.log(pred.clamp(min=self.eps))
            target_space = torch.log(target.clamp(min=self.eps))
        else:
            pred_space = pred
            target_space = target

        if mask is not None:
            valid = mask.bool()
            pred_space = torch.where(valid, pred_space, target_space)

        x = pred_space.flatten(0, 2)    # (B*T, 1, H, W)
        y = target_space.flatten(0, 2)  # (B*T, 1, H, W)

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
