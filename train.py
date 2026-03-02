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
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Optional

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
from training.prefetch import CPUPrefetchLoader



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

def _setup_logger(debug: bool = False, verbose: bool = False, log_file: str = "runs/train_verbose.log") -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger("focusmamba.train")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

    if verbose or debug:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        abs_log_path = str(log_path.resolve())
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_log_path
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(abs_log_path, mode="a")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
            )
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train FocusMamba — Metric Depth")
    parser.add_argument("--config", type=str, default="configs/experiments/tartanair_v2.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--verbose", action="store_true", help="Enable additional runtime logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--verbose-log-file",
        type=str,
        default="runs/train_verbose.log",
        help="Path for verbose/debug file logging",
    )
    args = parser.parse_args()

    logger = _setup_logger(debug=args.debug, verbose=args.verbose, log_file=args.verbose_log_file)

    # ── Distributed setup ────────────────────────────────────────────────────
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (local_rank == 0)  # only rank 0 does logging / checkpointing

    # Only rank 0 writes logs to avoid 4× duplicate output
    if not is_main:
        logging.disable(logging.CRITICAL)

    logger.info("Starting training script")
    logger.info(
        "Args: config=%s resume=%s device=%s verbose=%s debug=%s verbose_log_file=%s",
        args.config,
        args.resume,
        args.device,
        args.verbose,
        args.debug,
        args.verbose_log_file,
    )
    if args.verbose or args.debug:
        logger.info("Verbose/debug logs are being written to: %s", args.verbose_log_file)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    logger.debug("Loaded config:\n%s", pformat(cfg, sort_dicts=False))

    logger.info("Device: %s  |  world_size=%d  local_rank=%d", device, world_size, local_rank)

    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    use_fp16 = train_cfg.get("precision", "bf16") == "fp16" and device.type == "cuda"
    use_amp  = train_cfg.get("precision", "bf16") in ("bf16", "fp16") and device.type == "cuda"
    logger.info("Runtime settings: use_amp=%s use_fp16=%s precision=%s", use_amp, use_fp16, train_cfg.get("precision", "bf16"))
    logger.debug("Training config:\n%s", pformat(train_cfg, sort_dicts=False))
    logger.debug("Data config:\n%s", pformat(data_cfg, sort_dicts=False))

    # -----------------------------------------------------------------------
    # Build student model
    # -----------------------------------------------------------------------
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = cfg.get("model", {}).get("type", "mamba")
    logger.info("Student: %s | Parameters: %s", model_type, f"{n_params:,}")

    # torch.compile — fuses kernels and eliminates Python overhead.
    # Gives ~20-40 % throughput improvement on modern CUDA GPUs including GH200.
    if device.type == "cuda" and train_cfg.get("compile", True):
        try:
            model = torch.compile(model)
            logger.info("torch.compile OK")
        except Exception as e:
            logger.warning("torch.compile failed (%s), continuing without compilation", e)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        logger.info("DDP: world_size=%d local_rank=%d", world_size, local_rank)

    # -----------------------------------------------------------------------
    # Build teacher models (frozen, for distillation)
    # -----------------------------------------------------------------------
    active_teachers: Dict[str, nn.Module] = {}
    teacher_weights: Dict[str, float] = {}
    teacher_cfgs: List[dict] = cfg.get("teachers", [])

    # When a pre-built teacher cache exists (data.teacher_cache_dir is set) and
    # training.skip_live_teachers is true, skip loading the heavy teacher models
    # entirely.  The trainer will read pseudo-labels from disk; any rare cache
    # miss simply won't have a distillation term that step.
    _has_cache = bool(data_cfg.get("teacher_cache_dir"))
    _skip_live = train_cfg.get("skip_live_teachers", _has_cache)

    if _skip_live and _has_cache:
        logger.info("Teacher cache found — skipping live teacher loading (skip_live_teachers=True).")
        logger.info("Run tools/cache_teacher_labels.py to rebuild the cache if needed.")
    else:
        for t_cfg in teacher_cfgs:
            t_name = t_cfg.get("name", "")
            if not t_name or not t_cfg.get("enabled", True):
                continue
            w = float(t_cfg.get("weight", 1.0))
            try:
                teacher = build_teacher(t_name, t_cfg, device=str(device))
                active_teachers[t_name] = teacher
                teacher_weights[t_name] = w
                logger.info("Teacher registered: %s (weight=%s)", t_name, w)
            except Exception as e:
                logger.warning("Could not register teacher '%s': %s", t_name, e)

    if active_teachers:
        logger.info("Active teachers: %s", list(active_teachers.keys()))
    else:
        if not _has_cache:
            logger.info("No teachers loaded — training with GT depth only")

    # -----------------------------------------------------------------------
    # Build dataloaders
    # -----------------------------------------------------------------------
    dataset_type = data_cfg.get("dataset", "tartanair_v2")

    if dataset_type == "tartanair_v2":
        from dataloader.tartanair_v2 import TartanAirV2Dataset

        _teacher_cache_dir = data_cfg.get("teacher_cache_dir", None)
        if _teacher_cache_dir:
            logger.info("Teacher cache: %s", _teacher_cache_dir)
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
            teacher_cache_dir=_teacher_cache_dir,
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
            teacher_cache_dir=_teacher_cache_dir,
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
    # persistent_workers keeps worker processes alive between epochs, saving
    # the fork overhead (~1-3 s per epoch on disk-heavy datasets).
    # prefetch_factor lets workers queue up batches while the GPU is busy.
    _persistent = num_workers > 0
    _prefetch = 2 if num_workers > 0 else None

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        drop_last=True,
    ) if is_distributed else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        drop_last=False,
    ) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )

    # Compute batch count before optionally wrapping in CPUPrefetchLoader,
    # which may not implement __len__.
    n_train_batches = len(train_loader)

    # Optional CPU prefetch queue on top of DataLoader worker prefetch.
    # Useful on shared filesystems where producer jitter can still stall the
    # training loop despite prefetch_factor.
    if train_cfg.get("cpu_prefetch", True):
        cpu_prefetch_batches = int(train_cfg.get("cpu_prefetch_batches", 2))
        train_loader = CPUPrefetchLoader(train_loader, prefetch=cpu_prefetch_batches)
        val_loader = CPUPrefetchLoader(val_loader, prefetch=cpu_prefetch_batches)
    logger.info("Train clips: %d | Val clips: %d", len(train_dataset), len(val_dataset))
    logger.info(
        "Dataloader settings: batch_size=%d num_workers=%d persistent_workers=%s prefetch_factor=%s cpu_prefetch=%s",
        batch_size,
        num_workers,
        _persistent,
        _prefetch,
        train_cfg.get("cpu_prefetch", True),
    )

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
    # When using cached teacher depths (no live teachers), enable distillation
    # from the teacher configs in the YAML so CombinedLoss creates the
    # DistillationLoss module.  Cached depths are keyed by teacher name and
    # loaded by the DataLoader — no live teacher models are needed.
    if not active_teachers and _has_cache and teacher_cfgs:
        distillation_cfg["enabled"] = True
        distillation_cfg.setdefault("teachers", [
            {"name": t["name"], "weight": float(t.get("weight", 1.0)), "loss": "si_log"}
            for t in teacher_cfgs if t.get("enabled", True)
        ])
        logger.info("Distillation enabled from cached teacher depths: %s",
                    [t["name"] for t in distillation_cfg["teachers"]])
    criterion = CombinedLoss(cfg=loss_cfg, distillation_cfg=distillation_cfg)

    lr = train_cfg.get("learning_rate", 2e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # bf16 has fp32 exponent range; no loss scaling needed

    max_epochs = train_cfg.get("max_epochs", 50)
    warmup_steps = train_cfg.get("warmup_steps", 200)
    # n_train_batches is captured from the DataLoader that already has
    # DistributedSampler attached, so it already reflects the per-rank step
    # count (DistributedSampler.__len__ returns ceil(len(dataset)/world_size)).
    # No further division by world_size is needed.
    total_steps = max_epochs * max(n_train_batches, 1)

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
        logger.info("EMA enabled (decay=%s)", train_cfg["ema_decay"])

    # -----------------------------------------------------------------------
    # Logging & checkpoints
    # -----------------------------------------------------------------------
    log_dir = Path(train_cfg.get("log_dir", f"runs/{dataset_type}"))
    if is_main:
        writer = SummaryWriter(str(log_dir))
        writer.add_text("config", yaml.dump(cfg), 0)
    else:
        writer = None
    logger.info("TensorBoard logs -> %s (run: tensorboard --logdir %s)", log_dir, log_dir)

    log_img_every = train_cfg.get("log_images_every_n_steps", 50)
    log_img_max_B = train_cfg.get("log_images_max_batch", 2)
    step_log_every = int(train_cfg.get("step_log_every_n_steps", 0))
    if args.debug and step_log_every < 1:
        step_log_every = 1

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", f"checkpoints/{dataset_type}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if args.resume and os.path.isfile(args.resume):
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # Sanity-check: old checkpoints stored AbsRel (e.g. 0.083) under a
        # different key; a combined loss is always >> 0.01.
        if best_val_loss < 0.01:
            logger.warning(
                "Suspicious best_val_loss=%.5f from checkpoint (likely old AbsRel), resetting to inf",
                best_val_loss,
            )
            best_val_loss = float("inf")
        global_step = ckpt.get("global_step", 0)
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        logger.info("Resumed from epoch %d, best_val_loss=%.5f", start_epoch, best_val_loss)
    elif args.resume:
        logger.warning("Resume checkpoint not found: %s", args.resume)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    val_every = train_cfg.get("val_every_n_epochs", 5)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    t_start = time.time()
    logger.info(
        "Training loop: start_epoch=%d max_epochs=%d val_every=%d grad_clip=%.3f",
        start_epoch,
        max_epochs,
        val_every,
        grad_clip,
    )
    logger.info("Step logging: every_n_steps=%d", step_log_every)

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
        disable=not is_main,
        **_TQDM,
    )

    for epoch in epoch_bar:
        epoch_t0 = time.time()

        # Shuffle sampler for this epoch — required so each rank sees a
        # different shard on every epoch; no-op when running single-GPU.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Update curriculum degradation scale (no-op until degradation is wired in)
        deg_scale = curriculum.get_scale(epoch)
        if args.verbose or args.debug:
            logger.info("Epoch %d/%d started | global_step=%d | deg_scale=%.3f", epoch + 1, max_epochs, global_step, deg_scale)
        if is_main:
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
            grad_accum_steps=train_cfg.get("gradient_accumulation_steps", 1),
            logger=logger,
            step_log_every_n_steps=step_log_every,
            is_main=is_main,
        )
        logger.debug("Epoch %d finished train_one_epoch | avg_loss=%.6f | global_step=%d", epoch + 1, avg_loss, global_step)
        epoch_time = time.time() - epoch_t0

        # ETA
        elapsed = time.time() - t_start
        epochs_done = epoch - start_epoch + 1
        eta_s = elapsed / epochs_done * (max_epochs - epoch - 1)
        eta_m = eta_s / 60.0

        if is_main:
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
                is_main=is_main,
            )

            if ema is not None:
                ema.restore()

            val_loss = val_metrics.get("loss", float("inf"))
            abs_rel = val_metrics.get("abs_rel", float("inf"))
            delta1 = val_metrics.get("delta1", 0.0)
            if is_main:
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

            # Save best checkpoint (rank 0 only)
            if is_main and val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_ckpt(
                    ckpt_dir / "best.pt",
                    epoch, model, optimizer, scaler, scheduler,
                    global_step, best_val_loss, cfg, ema,
                )
                tqdm.write(f"  -> Best saved (val_loss={best_val_loss:.5f})")
                logger.info("New best checkpoint saved: val_loss=%.5f (epoch=%d)", best_val_loss, epoch + 1)

        # Save latest checkpoint every epoch (rank 0 only)
        if is_main:
            _save_ckpt(
                ckpt_dir / "latest.pt",
                epoch, model, optimizer, scaler, scheduler,
                global_step, best_val_loss, cfg, ema,
            )
        logger.debug("Saved latest checkpoint for epoch %d", epoch + 1)

    # -----------------------------------------------------------------------
    # Final profiling (rank 0 only)
    # -----------------------------------------------------------------------
    if is_main:
        profiler = LatencyProfiler(
            input_shape=(1, 3, 8, 256, 256),
            target_fps=30.0,
        )
        try:
            latency = profiler.measure(model, device)
            logger.info("Latency: %.1f FPS, %.1f ms/frame", latency["fps"], latency["ms_per_frame"])
            if not latency["meets_target"]:
                logger.warning("Does not meet 30 FPS target")
        except Exception as e:
            logger.warning("Latency profiling skipped: %s", e)

        writer.close()

    logger.info("Training complete. Best val loss: %.5f", best_val_loss)
    logger.info("Checkpoints in: %s", ckpt_dir)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
