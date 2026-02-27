#!/usr/bin/env python3
"""
FocusMamba Training Script — Metric Video Depth Estimation
============================================================

Multi-teacher distillation of degradation-robust metric depth.
Teachers run on clean frames, student runs on degraded frames.

Features:
    - Multi-teacher distillation with confidence weighting
    - Degradation curriculum scheduling
    - bf16 mixed precision with GradScaler
    - Gradient clipping with NaN guards
    - EMA of student weights
    - Linear warmup → cosine decay LR schedule
    - TensorBoard logging of losses, metrics, depth visualisations
    - Best-checkpoint saving based on val AbsRel

Usage:
    python train.py --config configs/experiments/distill_v1.yaml
    python train.py --config configs/base.yaml --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import yaml

from models import build_model
from models.teachers import build_teacher
from training.losses.combined import CombinedLoss
from training.trainer import (
    WarmupCosineScheduler,
    train_one_epoch,
    validate,
)
from training.curriculum import CurriculumScheduler
from training.ema import EMAModel
from training.callbacks.visualise_depth import log_depth_visualisation
from training.callbacks.latency_profiler import LatencyProfiler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train FocusMamba — Metric Depth")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_cfg = cfg.get("training", {})
    use_amp = train_cfg.get("precision", "bf16") == "bf16" and device.type == "cuda"

    # -----------------------------------------------------------------------
    # Build student model
    # -----------------------------------------------------------------------
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = cfg.get("model", {}).get("type", "mamba")
    print(f"Student: {model_type}  |  Parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # Build teacher models (frozen, for distillation)
    # -----------------------------------------------------------------------
    teachers: Dict[str, nn.Module] = {}
    distill_cfg = cfg.get("distillation", {})
    # YAML stores teachers as a list: [{name, weight, loss, enabled?, ...}, ...]
    teacher_cfgs = distill_cfg.get("teachers", [])
    for t_cfg in teacher_cfgs:
        t_name = t_cfg.get("name")
        if not t_name:
            continue
        if not t_cfg.get("enabled", True):
            continue
        try:
            teacher = build_teacher(t_name, t_cfg, device=str(device))
            teachers[t_name] = teacher
            print(f"Teacher loaded: {t_name}")
        except (NotImplementedError, Exception) as e:
            print(f"Warning: Could not load teacher '{t_name}': {e}")

    if teachers:
        print(f"Active teachers: {list(teachers.keys())}")
    else:
        print("No teachers loaded — training without distillation")

    # -----------------------------------------------------------------------
    # Build dataloaders
    # -----------------------------------------------------------------------
    # NOTE: Connect your dataset here. The datamodule expects the new
    # BaseDepthDataset interface returning dict with:
    #   'clean_frames' (B,C,T,H,W), 'degraded_frames' (B,C,T,H,W),
    #   'depth' (B,1,T,H,W), 'metadata' (dict)
    try:
        from data.datamodule import build_dataloaders
        loaders = build_dataloaders(cfg)
        train_loader = loaders["train"]
        val_loader = loaders.get("val")
    except NotImplementedError:
        print("WARNING: build_dataloaders not yet implemented.")
        print("Please implement data/datamodule.py to continue training.")
        print("Exiting.")
        return

    # -----------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    loss_cfg = train_cfg.get("loss", {})
    criterion = CombinedLoss(
        si_log_weight=loss_cfg.get("si_log_weight", 1.0),
        distillation_weight=loss_cfg.get("distillation_weight", 1.0),
        gradient_weight=loss_cfg.get("gradient_weight", 0.5),
        temporal_weight=loss_cfg.get("temporal_weight", 0.1),
        uncertainty_nll_weight=loss_cfg.get("uncertainty_nll_weight", 0.0),
        teacher_names=list(teachers.keys()) if teachers else [],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = train_cfg.get("max_epochs", 200) * max(len(train_loader), 1)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        total_steps=total_steps,
        min_lr=1e-6,
    )

    # -----------------------------------------------------------------------
    # Curriculum scheduler for degradation severity
    # -----------------------------------------------------------------------
    deg_cfg = cfg.get("degradation", {}).get("curriculum", {})
    curriculum = CurriculumScheduler(
        total_epochs=train_cfg.get("max_epochs", 200),
        warmup_epochs=deg_cfg.get("warmup_epochs", 10),
        schedule=deg_cfg.get("schedule", "linear"),
    )

    # -----------------------------------------------------------------------
    # EMA of student weights
    # -----------------------------------------------------------------------
    ema: Optional[EMAModel] = None
    if train_cfg.get("ema_decay", 0) > 0:
        ema = EMAModel(model, decay=train_cfg["ema_decay"])
        print(f"EMA enabled (decay={train_cfg['ema_decay']})")

    # -----------------------------------------------------------------------
    # Logging & checkpoints
    # -----------------------------------------------------------------------
    log_dir = Path(train_cfg.get("log_dir", "./runs")) / f"{model_type}_depth"
    writer = SummaryWriter(str(log_dir))
    writer.add_text("config", yaml.dump(cfg), 0)

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "./checkpoints")) / f"{model_type}_depth"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch = 0
    best_abs_rel = float("inf")
    global_step = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_abs_rel = ckpt.get("best_abs_rel", float("inf"))
        global_step = ckpt.get("global_step", 0)
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        print(f"Resumed from epoch {start_epoch}, best_abs_rel={best_abs_rel:.5f}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    max_epochs = train_cfg.get("max_epochs", 200)
    val_every = train_cfg.get("val_every_n_epochs", 5)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    t_start = time.time()

    for epoch in range(start_epoch, max_epochs):
        epoch_t0 = time.time()

        # Update curriculum degradation scale
        deg_scale = curriculum.get_scale(epoch)
        writer.add_scalar("train/degradation_scale", deg_scale, epoch)
        # TODO: pass deg_scale to dataloader's degradation pipeline
        # e.g. train_loader.dataset.degradation_pipeline.curriculum_scale = deg_scale

        # Train one epoch
        avg_loss, global_step = train_one_epoch(
            model=model,
            teachers=teachers if teachers else None,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            grad_clip=grad_clip,
            use_amp=use_amp,
            writer=writer,
            global_step=global_step,
            ema=ema,
        )
        epoch_time = time.time() - epoch_t0

        # ETA
        elapsed = time.time() - t_start
        epochs_done = epoch - start_epoch + 1
        eta_s = elapsed / epochs_done * (max_epochs - epoch - 1)
        eta_m = eta_s / 60.0

        print(
            f"Epoch {epoch + 1}/{max_epochs}  "
            f"train_loss={avg_loss:.4f}  "
            f"deg_scale={deg_scale:.2f}  "
            f"time={epoch_time:.1f}s  "
            f"ETA={eta_m:.1f}min"
        )
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        if val_loader is not None and (
            (epoch + 1) % val_every == 0 or epoch == max_epochs - 1
        ):
            # Use EMA weights for validation if available
            if ema is not None:
                ema.apply_shadow()

            val_metrics = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                writer=writer,
                epoch=epoch,
            )

            if ema is not None:
                ema.restore()

            abs_rel = val_metrics.get("abs_rel", float("inf"))
            delta1 = val_metrics.get("delta1", 0.0)
            print(
                f"  Val — loss={val_metrics['loss']:.4f}  "
                f"AbsRel={abs_rel:.5f}  "
                f"δ1={delta1:.4f}  "
                f"RMSE={val_metrics.get('rmse', 0):.4f}  "
                f"SI-log={val_metrics.get('si_log', 0):.5f}"
            )

            # Save best checkpoint (based on val AbsRel)
            if abs_rel < best_abs_rel:
                best_abs_rel = abs_rel
                save_dict = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_abs_rel": best_abs_rel,
                    "global_step": global_step,
                    "config": cfg,
                }
                if ema is not None:
                    save_dict["ema"] = ema.state_dict()
                torch.save(save_dict, ckpt_dir / "best.pt")
                print(f"  -> Best checkpoint saved (AbsRel={best_abs_rel:.5f})")

        # Save latest checkpoint
        save_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_abs_rel": best_abs_rel,
            "global_step": global_step,
            "config": cfg,
        }
        if ema is not None:
            save_dict["ema"] = ema.state_dict()
        torch.save(save_dict, ckpt_dir / "latest.pt")

    # -----------------------------------------------------------------------
    # Final profiling
    # -----------------------------------------------------------------------
    profiler = LatencyProfiler(
        input_shape=(1, 3, 8, 256, 256),
        target_fps=30.0,
    )
    try:
        latency = profiler.measure(model, device)
        print(f"\nLatency: {latency['fps']:.1f} FPS, {latency['ms_per_frame']:.1f} ms/frame")
        if not latency["meets_target"]:
            print("  WARNING: Does not meet 30 FPS target!")
    except Exception as e:
        print(f"Latency profiling skipped: {e}")

    writer.close()
    print(f"\nTraining complete. Best val AbsRel: {best_abs_rel:.5f}")


if __name__ == "__main__":
    main()
