#!/usr/bin/env python3
"""
FocusMamba Training Script — Metric Video Depth Estimation
============================================================

Multi-teacher distillation + GT-supervised training for degradation-robust
metric depth. Supports TartanAir v2 (with GT depth) and YouTube-VOS
(teacher pseudo-GT only).

Loss modes:
    - GT L1/SI-log loss against ground-truth depth (TartanAir)
    - Teacher distillation loss (optional, additive)
    - Both simultaneously when GT + teachers are available

Features:
    - Multi-teacher distillation with confidence weighting
    - bf16 mixed precision with GradScaler
    - Gradient clipping with NaN guards
    - EMA of student weights
    - Linear warmup → cosine decay LR schedule
    - TensorBoard logging of losses, metrics, depth image grids
    - tqdm progress bars (throttled for non-TTY / SLURM)
    - Best-checkpoint saving based on val loss or AbsRel

Usage:
    python train.py --config configs/experiments/tartanair_v2.yaml
    python train.py --config configs/experiments/trial_youtube_vos.yaml
    python train.py --config configs/experiments/tartanair_v2.yaml \\
        --resume checkpoints/tartanair_v2/latest.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from training.callbacks.latency_profiler import LatencyProfiler



# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_ckpt(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    scaler,
    scheduler,
    global_step: int,
    best_val_loss: float,
    cfg: dict,
    ema: Optional[EMAModel] = None,
) -> None:
    save_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    if ema is not None:
        save_dict["ema"] = ema.state_dict()
    torch.save(save_dict, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train FocusMamba — Metric Depth")
    parser.add_argument("--config", type=str, default="configs/experiments/tartanair_v2.yaml")
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
    print(f"Device: {device}")

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
    active_teachers: Dict[str, nn.Module] = {}
    teacher_weights: Dict[str, float] = {}
    teacher_cfgs: List[dict] = cfg.get("teachers", [])

    for t_cfg in teacher_cfgs:
        t_name = t_cfg.get("name", "")
        if not t_name or not t_cfg.get("enabled", True):
            continue
        w = float(t_cfg.get("weight", 1.0))
        try:
            teacher = build_teacher(t_name, t_cfg, device=str(device))
            active_teachers[t_name] = teacher
            teacher_weights[t_name] = w
            print(f"Teacher registered: {t_name}  (weight={w})")
        except Exception as e:
            print(f"Warning: could not register teacher '{t_name}': {e}")

    if active_teachers:
        print(f"Active teachers: {list(active_teachers.keys())}")
    else:
        print("No teachers loaded — training with GT depth only")

    # -----------------------------------------------------------------------
    # Build dataloaders
    # -----------------------------------------------------------------------
    data_cfg = cfg.get("data", {})
    dataset_type = data_cfg.get("dataset", "tartanair_v2")

    if dataset_type == "tartanair_v2":
        from dataloader.tartanair_v2 import TartanAirV2Dataset

        train_dataset = TartanAirV2Dataset(
            root=data_cfg.get("root", "/projects/b5dh/data/tartanair-v2"),
            num_frames=data_cfg.get("num_frames", 8),
            image_size=tuple(data_cfg.get("image_size", [256, 256])),
            max_trajectories=data_cfg.get("max_trajectories", None),
            clip_stride=data_cfg.get("clip_stride", 8),
            frame_stride=data_cfg.get("frame_stride", 1),
            split="train",
            val_fraction=data_cfg.get("val_fraction", 0.1),
            seed=data_cfg.get("seed", 42),
            difficulty=data_cfg.get("difficulty", "Data_easy"),
            camera=data_cfg.get("camera", "lcam_front"),
            max_depth=data_cfg.get("max_depth", 80.0),
            envs=data_cfg.get("envs", None),
        )
        val_dataset = TartanAirV2Dataset(
            root=data_cfg.get("root", "/projects/b5dh/data/tartanair-v2"),
            num_frames=data_cfg.get("num_frames", 8),
            image_size=tuple(data_cfg.get("image_size", [256, 256])),
            max_trajectories=data_cfg.get("max_trajectories", None),
            clip_stride=data_cfg.get("clip_stride", 8),
            frame_stride=data_cfg.get("frame_stride", 1),
            split="val",
            val_fraction=data_cfg.get("val_fraction", 0.1),
            seed=data_cfg.get("seed", 42),
            difficulty=data_cfg.get("difficulty", "Data_easy"),
            camera=data_cfg.get("camera", "lcam_front"),
            max_depth=data_cfg.get("max_depth", 80.0),
            envs=data_cfg.get("envs", None),
        )
    elif dataset_type == "youtube_vos":
        from dataloader.youtube_vos import YouTubeVOSDataset

        train_dataset = YouTubeVOSDataset(
            root=data_cfg.get("root", "data/youtube-vos/train_all_frames/JPEGImages"),
            num_frames=data_cfg.get("num_frames", 8),
            image_size=tuple(data_cfg.get("image_size", [256, 256])),
            max_videos=data_cfg.get("max_videos", None),
            clip_stride=data_cfg.get("clip_stride", 8),
            frame_stride=data_cfg.get("frame_stride", 1),
            split="train",
            val_fraction=data_cfg.get("val_fraction", 0.1),
            seed=data_cfg.get("seed", 42),
        )
        val_dataset = YouTubeVOSDataset(
            root=data_cfg.get("root", "data/youtube-vos/train_all_frames/JPEGImages"),
            num_frames=data_cfg.get("num_frames", 8),
            image_size=tuple(data_cfg.get("image_size", [256, 256])),
            max_videos=data_cfg.get("max_videos", None),
            clip_stride=data_cfg.get("clip_stride", 8),
            frame_stride=data_cfg.get("frame_stride", 1),
            split="val",
            val_fraction=data_cfg.get("val_fraction", 0.1),
            seed=data_cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type!r}. Use 'tartanair_v2' or 'youtube_vos'.")

    batch_size = train_cfg.get("batch_size", 2)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(f"Train clips: {len(train_dataset)}  |  Val clips: {len(val_dataset)}")

    # -----------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    loss_cfg = cfg.get("loss", {})
    distillation_cfg = cfg.get("distillation", {})
    if active_teachers and not distillation_cfg.get("enabled"):
        distillation_cfg["enabled"] = True
        distillation_cfg.setdefault("teachers", [
            {"name": n, "weight": teacher_weights.get(n, 1.0), "loss": "si_log"}
            for n in active_teachers.keys()
        ])
    criterion = CombinedLoss(cfg=loss_cfg, distillation_cfg=distillation_cfg)

    lr = train_cfg.get("learning_rate", 2e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    max_epochs = train_cfg.get("max_epochs", 50)
    warmup_steps = train_cfg.get("warmup_steps", 200)
    total_steps = max_epochs * max(len(train_loader), 1)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=1e-6,
    )

    # -----------------------------------------------------------------------
    # Curriculum scheduler for degradation severity (no-op for now)
    # -----------------------------------------------------------------------
    deg_cfg = cfg.get("degradation", {}).get("curriculum", {})
    if isinstance(deg_cfg, bool):
        deg_cfg = {}
    curriculum = CurriculumScheduler(
        total_epochs=max_epochs,
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
    log_dir = Path(train_cfg.get("log_dir", f"runs/{dataset_type}"))
    writer = SummaryWriter(str(log_dir))
    writer.add_text("config", yaml.dump(cfg), 0)
    print(f"TensorBoard logs -> {log_dir}  (run: tensorboard --logdir {log_dir})")

    log_img_every = train_cfg.get("log_images_every_n_steps", 50)
    log_img_max_B = train_cfg.get("log_images_max_batch", 2)

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", f"checkpoints/{dataset_type}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("best_abs_rel", float("inf")))
        global_step = ckpt.get("global_step", 0)
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.5f}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    val_every = train_cfg.get("val_every_n_epochs", 5)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    t_start = time.time()

    # tqdm settings — throttle in SLURM non-TTY
    _is_tty = sys.stdout.isatty()
    _TQDM = dict(
        file=sys.stdout,
        dynamic_ncols=False,
        ncols=100,
        ascii=True,
        mininterval=1.0 if _is_tty else 60.0,
    )

    epoch_bar = tqdm(
        range(start_epoch, max_epochs),
        desc="epochs",
        unit="epoch",
        initial=start_epoch,
        total=max_epochs,
        **_TQDM,
    )

    for epoch in epoch_bar:
        epoch_t0 = time.time()

        # Update curriculum degradation scale (no-op until degradation is wired in)
        deg_scale = curriculum.get_scale(epoch)
        writer.add_scalar("train/degradation_scale", deg_scale, epoch)

        # Train one epoch
        avg_loss, global_step = train_one_epoch(
            model=model,
            teachers=active_teachers if active_teachers else None,
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
            teacher_weights=teacher_weights if active_teachers else None,
            log_img_every=log_img_every,
            log_img_max_B=log_img_max_B,
        )
        epoch_time = time.time() - epoch_t0

        # ETA
        elapsed = time.time() - t_start
        epochs_done = epoch - start_epoch + 1
        eta_s = elapsed / epochs_done * (max_epochs - epoch - 1)
        eta_m = eta_s / 60.0

        tqdm.write(
            f"Epoch {epoch + 1}/{max_epochs}  "
            f"train_loss={avg_loss:.5f}  "
            f"deg_scale={deg_scale:.2f}  "
            f"time={epoch_time:.1f}s  "
            f"ETA={eta_m:.1f}min"
        )
        epoch_bar.set_postfix(train_loss=f"{avg_loss:.5f}")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        if (epoch + 1) % val_every == 0 or epoch == max_epochs - 1:
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
                teachers=active_teachers if active_teachers else None,
                teacher_weights=teacher_weights if active_teachers else None,
                log_img_max_B=log_img_max_B,
            )

            if ema is not None:
                ema.restore()

            val_loss = val_metrics.get("loss", float("inf"))
            abs_rel = val_metrics.get("abs_rel", float("inf"))
            delta1 = val_metrics.get("delta1", 0.0)
            tqdm.write(
                f"  Val — loss={val_loss:.5f}  "
                f"AbsRel={abs_rel:.5f}  "
                f"d1={delta1:.4f}  "
                f"RMSE={val_metrics.get('rmse', 0):.4f}  "
                f"SI-log={val_metrics.get('si_log', 0):.5f}"
            )
            epoch_bar.set_postfix(
                train_loss=f"{avg_loss:.5f}",
                val_loss=f"{val_loss:.5f}",
            )
            writer.add_scalar("val/epoch_loss", val_loss, epoch)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_ckpt(
                    ckpt_dir / "best.pt",
                    epoch, model, optimizer, scaler, scheduler,
                    global_step, best_val_loss, cfg, ema,
                )
                tqdm.write(f"  -> Best saved (val_loss={best_val_loss:.5f})")

        # Save latest checkpoint every epoch
        _save_ckpt(
            ckpt_dir / "latest.pt",
            epoch, model, optimizer, scaler, scheduler,
            global_step, best_val_loss, cfg, ema,
        )

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
    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")
    print(f"Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()
