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

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
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
) -> tuple[float, int]:
    """Run one training epoch.

    For each batch:
        1. Student processes DEGRADED frames → predicted depth
        2. Teachers (frozen) process CLEAN frames → teacher depths
        3. Compute combined loss (distillation + others)
        4. Backprop to student only

    Returns:
        (avg_loss, updated_global_step)
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", unit="it", leave=False):
        degraded_frames = batch["degraded_frames"].to(device)  # (B, C, T, H, W)
        clean_frames = batch["clean_frames"].to(device)        # (B, C, T, H, W)
        gt_depth = batch.get("depth")
        if gt_depth is not None:
            gt_depth = gt_depth.to(device)  # (B, 1, T, H, W)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            # Student forward on degraded input
            student_outputs = model(degraded_frames)
            student_depth = student_outputs["depth"]

            # Guard: skip if non-finite
            if not torch.isfinite(student_depth).all():
                optimizer.zero_grad(set_to_none=True)
                continue

            # Teacher forward on clean input (no grad)
            teacher_depths = {}
            if teachers:
                with torch.no_grad():
                    for name, teacher in teachers.items():
                        teacher_depths[name] = teacher.predict(clean_frames)

            # Compute combined loss
            losses = criterion(
                student_outputs=student_outputs,
                teacher_depths=teacher_depths if teacher_depths else None,
                gt_depth=gt_depth,
            )

        total_loss = losses["total"]
        if not torch.isfinite(total_loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        # NaN gradient guard
        grads_ok = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if not grads_ok:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # EMA update
        if ema is not None:
            ema.update()

        running_loss += total_loss.item()
        n_batches += 1
        global_step += 1

        # TensorBoard logging
        if global_step % 50 == 0:
            writer.add_scalar("train/loss_total", total_loss.item(), global_step)
            for k, v in losses.items():
                if k != "total" and isinstance(v, torch.Tensor):
                    writer.add_scalar(f"train/{k}", v.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

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
) -> Dict[str, float]:
    """Run validation and compute depth metrics."""
    model.eval()
    total_loss = 0.0
    metric_accum: Dict[str, float] = {}
    n_batches = 0

    for batch in tqdm(loader, desc=f"Val {epoch}", unit="it", leave=False):
        # For validation, use clean frames (no degradation)
        frames = batch.get("clean_frames", batch.get("degraded_frames")).to(device)
        gt_depth = batch.get("depth")
        if gt_depth is not None:
            gt_depth = gt_depth.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(frames)
            pred_depth = outputs["depth"]

            if gt_depth is not None:
                losses = criterion(
                    student_outputs=outputs,
                    gt_depth=gt_depth,
                )
                if torch.isfinite(losses["total"]):
                    total_loss += losses["total"].item()

        # Metrics
        if gt_depth is not None:
            m = compute_depth_metrics(pred_depth.float(), gt_depth.float())
            for k, v in m.items():
                metric_accum[k] = metric_accum.get(k, 0.0) + v
            n_batches += 1

    if n_batches == 0:
        return {"loss": float("inf"), "abs_rel": float("inf"), "delta1": 0.0}

    avg_metrics = {k: v / n_batches for k, v in metric_accum.items()}
    avg_metrics["loss"] = total_loss / n_batches

    # Log to TensorBoard
    writer.add_scalar("val/loss", avg_metrics["loss"], epoch)
    for k, v in avg_metrics.items():
        if k != "loss":
            writer.add_scalar(f"val/{k}", v, epoch)

    return avg_metrics
