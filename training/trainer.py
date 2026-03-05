"""
Trainer — Raw PyTorch Training Loop for Metric Video Depth
============================================================

Full training loop with:
    - Multi-teacher distillation (teachers run on clean, student on degraded)
    - bf16 AMP with GradScaler
    - Gradient clipping with NaN guards
    - EMA of student weights
    - Curriculum degradation scheduling
    - TensorBoard logging
    - Best-checkpoint saving by validation AbsRel
    - Warmup + cosine decay LR schedule

Usage:
    python train.py --config configs/experiments/distill_v1.yaml
"""

from __future__ import annotations

from contextlib import nullcontext
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.losses.combined import CombinedLoss
from training.curriculum import CurriculumScheduler
from training.ema import EMAModel


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Depth metrics for validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute standard depth estimation metrics.

    Args:
        pred: (B, 1, T, H, W) predicted metric depth.
        gt:   (B, 1, T, H, W) ground-truth metric depth.
        mask: (B, 1, T, H, W) optional validity mask.

    Returns:
        Dict with AbsRel, SqRel, RMSE, delta1, delta2, delta3, SI-log.
    """
    if mask is not None:
        pred = pred[mask.bool()]
        gt = gt[mask.bool()]
    else:
        pred = pred.flatten()
        gt = gt.flatten()

    # Filter invalid values
    valid = (gt > 1e-3) & (pred > 1e-3) & torch.isfinite(pred) & torch.isfinite(gt)
    pred = pred[valid]
    gt = gt[valid]

    if pred.numel() == 0:
        return {
            "abs_rel": float("inf"), "sq_rel": float("inf"),
            "rmse": float("inf"), "si_log": float("inf"),
            "delta1": 0.0, "delta2": 0.0, "delta3": 0.0,
        }

    # Absolute Relative Error
    abs_rel = ((pred - gt).abs() / gt).mean().item()

    # Squared Relative Error
    sq_rel = (((pred - gt) ** 2) / gt).mean().item()

    # RMSE
    rmse = ((pred - gt) ** 2).mean().sqrt().item()

    # SI-log
    log_diff = torch.log(pred) - torch.log(gt)
    si_log = (log_diff ** 2).mean().item() - (log_diff.mean().item()) ** 2

    # Delta thresholds
    ratio = torch.max(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "si_log": si_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


@torch.no_grad()
def _normalize_relative_depth(
    depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    log_space: bool = True,
    q_low: float = 0.02,
    q_high: float = 0.98,
    clamp: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-clip robust normalization to relative depth in [0, 1]."""
    base = torch.log(depth.clamp(min=eps)) if log_space else depth

    valid = torch.isfinite(base)
    if mask is not None:
        valid = valid & mask.bool()

    out = torch.zeros_like(base)
    b = base.shape[0]
    for i in range(b):
        vi = valid[i]
        if vi.sum() < 8:
            continue
        vals = base[i][vi]
        lo = torch.quantile(vals, q_low)
        hi = torch.quantile(vals, q_high)
        denom = (hi - lo).clamp(min=eps)
        x = (base[i] - lo) / denom
        if clamp:
            x = x.clamp(0.0, 1.0)
        out[i] = torch.where(vi, x, torch.zeros_like(x))
    return out


@torch.no_grad()
def compute_relative_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute relative-depth metrics on robustly normalized depth."""
    if mask is None:
        mask = gt > 0
    else:
        mask = mask.bool()

    pred_rel = _normalize_relative_depth(pred, mask)
    gt_rel = _normalize_relative_depth(gt, mask)

    valid = mask & torch.isfinite(pred_rel) & torch.isfinite(gt_rel)
    if not valid.any():
        return {
            "rel_l1": float("inf"),
            "rel_rmse": float("inf"),
            "rel_delta1": 0.0,
            # Backward-compatible aliases used by train.py summary logs.
            "abs_rel": float("inf"),
            "rmse": float("inf"),
            "delta1": 0.0,
        }

    p = pred_rel[valid]
    g = gt_rel[valid]
    diff = p - g
    rel_l1 = diff.abs().mean().item()
    rel_rmse = (diff ** 2).mean().sqrt().item()
    ratio = torch.max((p + 1e-6) / (g + 1e-6), (g + 1e-6) / (p + 1e-6))
    rel_delta1 = (ratio < 1.25).float().mean().item()

    return {
        "rel_l1": rel_l1,
        "rel_rmse": rel_rmse,
        "rel_delta1": rel_delta1,
        "abs_rel": rel_l1,
        "rmse": rel_rmse,
        "delta1": rel_delta1,
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    teachers: Optional[Dict[str, nn.Module]],
    loader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    writer: SummaryWriter,
    global_step: int,
    ema: Optional[EMAModel] = None,
    teacher_weights: Optional[Dict[str, float]] = None,
    log_img_every: int = 50,
    log_img_max_B: int = 2,
    grad_accum_steps: int = 1,
    logger: Optional[logging.Logger] = None,
    step_log_every_n_steps: int = 0,
    is_main: bool = True,
) -> tuple[float, int]:
    """Run one training epoch.

    Supports two data modes:
        1. No degradation: batch has ``frames`` (clean RGB) + optional ``depth`` (GT).
           Student and teachers both run on ``frames``.
        2. Degradation: batch has ``degraded_frames`` + ``clean_frames``.
           Student runs on degraded, teachers on clean.

    Teacher pseudo-labels are loaded from ``batch["cached_teacher_depths"]`` when
    present (pre-cached with ``tools/cache_teacher_labels.py``), falling back to
    live inference only for any teacher not found in the cache.

    When ``grad_accum_steps > 1`` the optimizer is stepped only every
    ``grad_accum_steps`` micro-batches, giving a larger effective batch size
    without increasing VRAM usage.

    Returns:
        (avg_loss, updated_global_step)
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    def _optimizer_step() -> bool:
        """Execute one optimizer/scheduler/EMA update from accumulated grads."""
        scaler.unscale_(optimizer)

        grads_ok = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if not grads_ok:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return False

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update()
        return True

    for micro_step, batch in enumerate(tqdm(loader, desc="Train", unit="it", leave=False)):
        is_accum_step = ((micro_step + 1) % grad_accum_steps != 0)
        use_no_sync = is_accum_step and hasattr(model, "no_sync")
        sync_ctx = model.no_sync() if use_no_sync else nullcontext()
        target_mode = getattr(criterion, "target_mode", "metric")

        # ── Resolve input frames ────────────────────────────────────────
        if "degraded_frames" in batch:
            student_input = batch["degraded_frames"].to(device)
            teacher_input = batch["clean_frames"].to(device)
        else:
            # No degradation — both see the same (clean) frames
            student_input = batch["frames"].to(device)
            teacher_input = student_input

        gt_depth = batch.get("depth")
        if gt_depth is not None:
            gt_depth = gt_depth.to(device)  # (B, 1, T, H, W)

        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)

        # Mark the start of a new CUDAGraph step so that torch.compile
        # (mode="reduce-overhead") knows it is safe to overwrite the output
        # buffers recorded from the previous iteration.  Without this call,
        # pin_memory DataLoader buffers are recycled before the backward pass
        # has finished reading them, causing the "overwritten by a subsequent
        # run" RuntimeError.
        # torch.compiler.cudagraph_mark_step_begin()
        with sync_ctx:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                # Student forward
                student_outputs = model(student_input)
                student_depth_metric = student_outputs["depth"]
                if target_mode == "relative":
                    student_depth = student_outputs.get("depth_relative", student_depth_metric)
                else:
                    student_depth = student_depth_metric

                # Guard: skip if non-finite
                if not torch.isfinite(student_depth).all():
                    if logger is not None:
                        logger.warning("step=%d non-finite student depth, skipping", global_step)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                # ── Teacher depths ────────────────────────────────────────
                # Priority 1: cached pseudo-labels from disk (fast, no GPU cost).
                # Priority 2: live teacher inference (slow, kept as fallback).
                teacher_depths: Dict[str, torch.Tensor] = {}

                # Load from cache if the batch carries pre-computed depths
                cached_batch: Optional[Dict[str, torch.Tensor]] = batch.get("cached_teacher_depths")
                if cached_batch is not None:
                    for t_name, td in cached_batch.items():
                        # td: (B, 1, T, H, W) from collation of per-sample (1, T, H, W) tensors
                        teacher_depths[t_name] = td.to(device, non_blocking=True)

                # Fall back to live inference for any teacher missing from cache
                if teachers:
                    missing = [n for n in teachers if n not in teacher_depths]
                    if missing:
                        with torch.no_grad():
                            for name in missing:
                                try:
                                    teacher_depths[name] = teachers[name].predict(teacher_input)
                                except Exception:
                                    pass

                # Compute combined loss; scale by 1/grad_accum_steps so that
                # accumulated gradients equal the true gradient for the macro-batch.
                losses = criterion(
                    student_outputs=student_outputs,
                    teacher_depths=teacher_depths if teacher_depths else None,
                    gt_depth=gt_depth,
                    mask=mask,
                )

            total_loss = losses["total"]
            if not torch.isfinite(total_loss):
                if logger is not None:
                    logger.warning("step=%d non-finite loss=%s, skipping", global_step, total_loss.item())
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaled_loss = total_loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

        # Only update weights at the end of each accumulation window
        if not is_accum_step:
            if not _optimizer_step():
                continue

        running_loss += total_loss.item()
        n_batches += 1
        global_step += 1

        if logger is not None and step_log_every_n_steps > 0 and (global_step % step_log_every_n_steps == 0):
            current_lr = optimizer.param_groups[0]["lr"]
            component_items = []
            for k, v in losses.items():
                if k == "total":
                    continue
                if isinstance(v, torch.Tensor):
                    component_items.append(f"{k}={float(v.detach().item()):.6f}")
                else:
                    component_items.append(f"{k}={float(v):.6f}")
            component_str = " | ".join(component_items) if component_items else "no_components"
            # Build a string with all loss components (including "total")
            loss_items = []
            for k, v in losses.items():
                try:
                    val = float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
                except Exception:
                    val = float("nan")
                loss_items.append(f"{k}={val:.6f}")
            loss_str = " | ".join(loss_items)

            logger.debug(
                "step=%d micro_step=%d lr=%.8f accum=%d/%d | %s",
                global_step,
                micro_step,
                current_lr,
                (micro_step % grad_accum_steps) + 1,
                grad_accum_steps,
                loss_str,
            )

        # TensorBoard logging — scalars
        if is_main and global_step % 50 == 0:
            writer.add_scalar("train/loss_total", total_loss.item(), global_step)
            for k, v in losses.items():
                if k != "total" and isinstance(v, torch.Tensor):
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        # TensorBoard logging — image grids (matching test_training.py style)
        if is_main and global_step % log_img_every == 0:
            with torch.no_grad():
                from training.callbacks.visualise_depth import colorise_depth
                pred_d = student_depth.detach()
                if target_mode == "relative":
                    vis_mask = mask if mask is not None else (gt_depth > 0 if gt_depth is not None else None)
                    gt_vis = _normalize_relative_depth(gt_depth, vis_mask) if gt_depth is not None else None
                    teacher_vis = {
                        t_name: _normalize_relative_depth(td, vis_mask)
                        for t_name, td in teacher_depths.items()
                    }
                else:
                    gt_vis = gt_depth
                    teacher_vis = teacher_depths

                B_vis = min(pred_d.shape[0], log_img_max_B)
                rows = []
                for b in range(B_vis):
                    t_mid = pred_d.shape[2] // 2
                    rgb = student_input[b, :, t_mid].cpu()  # (3, H, W)
                    s_d = pred_d[b, 0, t_mid].cpu()
                    col = [rgb, colorise_depth(s_d)]
                    if gt_vis is not None:
                        gt_d = gt_vis[b, 0, t_mid].cpu()
                        col.append(colorise_depth(gt_d))
                    for t_name, td in teacher_vis.items():
                        col.append(colorise_depth(td[b, 0, t_mid].cpu()))
                    rows.append(torch.cat(col, dim=1))
                grid = torch.cat(rows, dim=2).clamp(0.0, 1.0)
                writer.add_image("train/depth_grid", grid, global_step)

                # Per-teacher and student depth scale
                writer.add_scalar("mean_depth/student", pred_d.mean().item(), global_step)
                writer.add_scalar("mean_depth/student_metric", student_depth_metric.detach().mean().item(), global_step)
                for t_name, td in teacher_vis.items():
                    writer.add_scalar(f"mean_depth/teacher_{t_name}", td.mean().item(), global_step)
                if gt_vis is not None:
                    writer.add_scalar("mean_depth/gt", gt_vis.mean().item(), global_step)

    # Flush partial accumulation at epoch end so tail micro-batches are not dropped.
    if n_batches > 0 and (n_batches % grad_accum_steps != 0):
        has_grad = any(p.grad is not None for p in model.parameters())
        if has_grad:
            _optimizer_step()
            if logger is not None:
                logger.debug(
                    "Flushed partial accumulation at epoch end: remainder=%d",
                    n_batches % grad_accum_steps,
                )

    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss, global_step


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: CombinedLoss,
    device: torch.device,
    use_amp: bool,
    writer: SummaryWriter,
    epoch: int,
    teachers: Optional[Dict[str, nn.Module]] = None,
    teacher_weights: Optional[Dict[str, float]] = None,
    log_img_max_B: int = 2,
    is_main: bool = True,
) -> Dict[str, float]:
    """Run validation and compute depth metrics."""
    model.eval()
    total_loss = 0.0
    metric_accum: Dict[str, float] = {}
    n_batches = 0
    logged_image = False

    for batch in tqdm(loader, desc=f"Val {epoch}", unit="it", leave=False):
        target_mode = getattr(criterion, "target_mode", "metric")
        # Resolve input frames (same logic as train_one_epoch)
        if "clean_frames" in batch:
            frames = batch["clean_frames"].to(device)
        else:
            frames = batch["frames"].to(device)

        gt_depth = batch.get("depth")
        if gt_depth is not None:
            gt_depth = gt_depth.to(device)

        # torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(frames)
            pred_depth_metric = outputs["depth"]
            if target_mode == "relative":
                pred_depth = outputs.get("depth_relative", pred_depth_metric)
            else:
                pred_depth = pred_depth_metric

            # Teacher inference for distillation validation
            teacher_depths = {}
            if teachers:
                for name, teacher in teachers.items():
                    try:
                        teacher_depths[name] = teacher.predict(frames)
                    except Exception:
                        pass

            if gt_depth is not None:
                losses = criterion(
                    student_outputs=outputs,
                    teacher_depths=teacher_depths if teacher_depths else None,
                    gt_depth=gt_depth,
                )
                if torch.isfinite(losses["total"]):
                    total_loss += losses["total"].item()
            elif teacher_depths:
                losses = criterion(
                    student_outputs=outputs,
                    teacher_depths=teacher_depths,
                )
                if torch.isfinite(losses["total"]):
                    total_loss += losses["total"].item()

        # Metrics (against GT if available)
        if gt_depth is not None:
            if target_mode == "relative":
                val_mask = (gt_depth > 0)
                m = compute_relative_metrics(pred_depth.float(), gt_depth.float(), mask=val_mask)
            else:
                m = compute_depth_metrics(pred_depth.float(), gt_depth.float())
            for k, v in m.items():
                metric_accum[k] = metric_accum.get(k, 0.0) + v
            n_batches += 1

            # Log one image grid per validation
            if is_main and not logged_image:
                from training.callbacks.visualise_depth import colorise_depth
                B_vis = min(pred_depth.shape[0], log_img_max_B)
                if target_mode == "relative":
                    vis_mask = gt_depth > 0
                    gt_vis = _normalize_relative_depth(gt_depth, vis_mask)
                    teacher_vis = {
                        t_name: _normalize_relative_depth(td, vis_mask)
                        for t_name, td in teacher_depths.items()
                    }
                else:
                    gt_vis = gt_depth
                    teacher_vis = teacher_depths
                rows = []
                for b in range(B_vis):
                    t_mid = pred_depth.shape[2] // 2
                    rgb = frames[b, :, t_mid].cpu()
                    s_d = pred_depth[b, 0, t_mid].cpu()
                    col = [rgb, colorise_depth(s_d)]
                    gt_d = gt_vis[b, 0, t_mid].cpu()
                    col.append(colorise_depth(gt_d))
                    for t_name, td in teacher_vis.items():
                        col.append(colorise_depth(td[b, 0, t_mid].cpu()))
                    rows.append(torch.cat(col, dim=1))
                grid = torch.cat(rows, dim=2).clamp(0.0, 1.0)
                writer.add_image("val/depth_grid", grid, epoch)
                logged_image = True
        elif teacher_depths:
            # No GT — count batches by teacher availability
            n_batches += 1

    if n_batches == 0:
        return {"loss": float("inf"), "abs_rel": float("inf"), "delta1": 0.0}

    avg_metrics = {k: v / n_batches for k, v in metric_accum.items()}
    avg_metrics["loss"] = total_loss / n_batches

    # Aggregate metrics across all DDP ranks so rank-0 sees the global average
    if dist.is_available() and dist.is_initialized():
        for key in list(avg_metrics.keys()):
            t = torch.tensor(avg_metrics[key], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg_metrics[key] = t.item()

    # Log to TensorBoard (rank 0 only)
    if is_main:
        writer.add_scalar("val/loss", avg_metrics["loss"], epoch)
        for k, v in avg_metrics.items():
            if k != "loss":
                writer.add_scalar(f"val/{k}", v, epoch)

    return avg_metrics
