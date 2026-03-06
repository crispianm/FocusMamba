#!/usr/bin/env python3
"""
FocusMamba Training Script — Metric Video Depth Estimation
============================================================

Multi-teacher distillation + GT-supervised training for degradation-robust
metric depth. Supports TartanAir v2 (with GT depth) and YouTube-VOS
(teacher pseudo-GT only).

Loss modes:
    - VDA-style GT supervision: SSI + TGM (TartanAir)
    - Teacher distillation loss (optional novelty term)
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
import math
import logging
import os
import sys
import time
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
from training.logging_utils import (
    JsonlMetricLogger,
    resolve_run_artifacts,
    setup_logger,
    write_run_metadata,
)
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
    best_selection_metric: str,
    best_selection_value: float,
    cfg: dict,
    ema: Optional[EMAModel] = None,
    model_is_ema: bool = False,
) -> None:
    save_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "model_is_ema": bool(model_is_ema),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_selection_metric": best_selection_metric,
        "best_selection_value": best_selection_value,
        "config": cfg,
    }
    if ema is not None:
        save_dict["ema"] = ema.state_dict()
    torch.save(save_dict, path)

def _name_matches_prefix(name: str, prefix: str) -> bool:
    p = prefix.strip()
    if not p:
        return False
    current = name
    while True:
        if current == p or current.startswith(p + "."):
            return True
        if not current.startswith("module."):
            return False
        current = current[len("module.") :]


def _normalize_selection_metric(metric: str | None) -> str:
    key = str(metric or "loss").strip().lower()
    if key in ("val_loss", "validation_loss"):
        return "loss"
    return key


def _resolve_selection_metric(train_cfg: dict, loss_cfg: dict) -> str:
    explicit = train_cfg.get("selection_metric")
    if explicit is not None:
        metric = _normalize_selection_metric(explicit)
    else:
        target_mode = str(loss_cfg.get("training_target", "metric")).lower()
        if target_mode == "metric":
            metric = "abs_rel"
        elif target_mode == "relative":
            metric = "rel_l1"
        else:
            metric = "loss"
    valid_metrics = {
        "loss",
        "abs_rel",
        "sq_rel",
        "rmse",
        "si_log",
        "delta1",
        "delta2",
        "delta3",
        "rel_l1",
        "rel_rmse",
        "rel_delta1",
    }
    if metric not in valid_metrics:
        valid_str = ", ".join(sorted(valid_metrics))
        raise ValueError(f"Unsupported training.selection_metric={metric!r}. Valid options: {valid_str}")
    return metric


def _selection_metric_higher_is_better(metric: str) -> bool:
    return metric in {"delta1", "delta2", "delta3", "rel_delta1"}


def _initial_best_selection_value(metric: str) -> float:
    return float("-inf") if _selection_metric_higher_is_better(metric) else float("inf")


def _extract_selection_value(val_metrics: dict, selection_metric: str) -> float:
    if selection_metric not in val_metrics:
        available = ", ".join(sorted(val_metrics.keys()))
        raise KeyError(
            f"Selection metric '{selection_metric}' not found in validation metrics. "
            f"Available metrics: {available}"
        )
    return float(val_metrics[selection_metric])


def _is_better_selection_value(metric: str, current: float, best: float) -> bool:
    if _selection_metric_higher_is_better(metric):
        return current > best
    return current < best


def _scheduled_lr_for_group(scheduler: WarmupCosineScheduler, base_lr: float) -> float:
    step = max(int(getattr(scheduler, "last_epoch", 0)), 0)
    if step < scheduler.warmup_steps:
        scale = step / max(1, scheduler.warmup_steps)
    else:
        progress = (step - scheduler.warmup_steps) / max(
            1, scheduler.total_steps - scheduler.warmup_steps
        )
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(float(scheduler.min_lr), base_lr * scale)


def _unfreeze_backbone_group(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    train_cfg: dict,
    logger: logging.Logger,
) -> int:
    params = []
    n_unfrozen = 0
    for name, param in model.named_parameters():
        if any(
            _name_matches_prefix(name, p)
            for p in ("model.backbone", "backbone", "model.pretrained", "pretrained")
        ):
            if not param.requires_grad:
                param.requires_grad = True
                params.append(param)
                n_unfrozen += param.numel()

    if not params:
        logger.info("Scheduled backbone unfreeze requested, but backbone is already trainable.")
        return 0

    base_lr = float(train_cfg.get("learning_rate", 2e-4))
    backbone_mult = float(train_cfg.get("backbone_lr_multiplier", 1.0))
    group_base_lr = base_lr * backbone_mult
    current_lr = _scheduled_lr_for_group(scheduler, group_base_lr)
    optimizer.add_param_group(
        {
            "params": params,
            "lr": current_lr,
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
            "initial_lr": group_base_lr,
        }
    )
    scheduler.base_lrs.append(group_base_lr)
    logger.info(
        "Backbone unfrozen: params=%s base_lr=%.8f current_lr=%.8f (multiplier=%.3f)",
        f"{n_unfrozen:,}",
        group_base_lr,
        current_lr,
        backbone_mult,
    )
    return n_unfrozen


def _apply_freeze_cfg(model: nn.Module, train_cfg: dict, logger: logging.Logger) -> None:
    """Apply optional module freezing rules from config."""
    freeze_prefixes = [str(p) for p in train_cfg.get("freeze_prefixes", [])]
    unfreeze_prefixes = [str(p) for p in train_cfg.get("unfreeze_prefixes", [])]

    # Convenience flags for tiny dual-head VDA variants.
    if bool(train_cfg.get("freeze_backbone", False)):
        freeze_prefixes.extend(["model.backbone", "backbone", "model.pretrained", "pretrained"])
    if bool(train_cfg.get("freeze_relative_head", False)):
        freeze_prefixes.extend(["model.relative_head", "relative_head"])
    if bool(train_cfg.get("freeze_metric_head", False)):
        freeze_prefixes.extend(["model.metric_calibrator", "metric_calibrator"])

    if not freeze_prefixes and not unfreeze_prefixes:
        return

    frozen = 0
    unfrozen = 0

    for name, param in model.named_parameters():
        if any(_name_matches_prefix(name, p) for p in freeze_prefixes):
            if param.requires_grad:
                param.requires_grad = False
                frozen += param.numel()
        if any(_name_matches_prefix(name, p) for p in unfreeze_prefixes):
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Freeze config applied: frozen=%s unfrozen=%s trainable=%s/%s params",
        f"{frozen:,}",
        f"{unfrozen:,}",
        f"{trainable:,}",
        f"{total:,}",
    )
    if freeze_prefixes:
        logger.info("  freeze_prefixes=%s", freeze_prefixes)
    if unfreeze_prefixes:
        logger.info("  unfreeze_prefixes=%s", unfreeze_prefixes)


def _build_optimizer(model: nn.Module, train_cfg: dict, logger: logging.Logger) -> torch.optim.Optimizer:
    """Build AdamW with optional LR multipliers for backbone / metric head."""
    lr = float(train_cfg.get("learning_rate", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    metric_mult = float(train_cfg.get("metric_head_lr_multiplier", 1.0))
    backbone_mult = float(train_cfg.get("backbone_lr_multiplier", 1.0))

    groups: Dict[float, list] = {}
    counts: Dict[float, int] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scale = 1.0
        if any(_name_matches_prefix(name, p) for p in ("model.metric_calibrator", "metric_calibrator")):
            scale *= metric_mult
        elif any(
            _name_matches_prefix(name, p)
            for p in ("model.backbone", "backbone", "model.pretrained", "pretrained")
        ):
            scale *= backbone_mult
        groups.setdefault(scale, []).append(param)
        counts[scale] = counts.get(scale, 0) + param.numel()

    if not groups:
        raise RuntimeError(
            "No trainable parameters found after applying freeze config. "
            "Check training.freeze_* / freeze_prefixes settings."
        )

    param_groups = []
    for scale, params in sorted(groups.items(), key=lambda kv: kv[0]):
        param_groups.append({"params": params, "lr": lr * scale})
        logger.info(
            "Optimizer group: lr=%.8f scale=%.3f params=%s",
            lr * scale,
            scale,
            f"{counts[scale]:,}",
        )

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


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
    args = parser.parse_args()

    # Load config first so verbose/debug log file path is controlled by YAML.
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg.get("training", {})
    artifacts = resolve_run_artifacts(
        train_cfg,
        default_log_dir=f"runs/{Path(args.config).stem}",
    )
    logger = setup_logger(
        "focusmamba.train",
        debug=args.debug,
        log_file=artifacts.verbose_log_file,
        use_rich=True,
    )

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
        "Args: config=%s resume=%s device=%s verbose=%s debug=%s",
        args.config,
        args.resume,
        args.device,
        args.verbose,
        args.debug,
    )
    if args.verbose or args.debug:
        logger.info("Verbose/debug logs are being written to: %s", artifacts.verbose_log_file)

    logger.debug("Loaded config:\n%s", pformat(cfg, sort_dicts=False))

    logger.info("Device: %s  |  world_size=%d  local_rank=%d", device, world_size, local_rank)

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
    _apply_freeze_cfg(model, train_cfg, logger)
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

    # Video-Depth-Anything and FocusMamba-v2 (DPT temporal head) have a few
    # parameters that can be outside the active loss path. DDP must track
    # unused parameters for these model families.
    ddp_find_unused_cfg = train_cfg.get("ddp_find_unused_parameters", None)
    if ddp_find_unused_cfg is None:
        ddp_find_unused = model_type in ("video_depth_anything", "mamba")
    else:
        ddp_find_unused = bool(ddp_find_unused_cfg)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
        )
        logger.info(
            "DDP: world_size=%d local_rank=%d find_unused_parameters=%s",
            world_size,
            local_rank,
            ddp_find_unused,
        )

    # -----------------------------------------------------------------------
    # Build teacher models (frozen, for distillation)
    # -----------------------------------------------------------------------
    active_teachers: Dict[str, nn.Module] = {}
    teacher_weights: Dict[str, float] = {}
    teacher_cfgs: List[dict] = cfg.get("teachers", [])
    distillation_requested = bool(cfg.get("distillation", {}).get("enabled", False))

    # When a pre-built teacher cache exists (data.teacher_cache_dir is set) and
    # training.skip_live_teachers is true, skip loading the heavy teacher models
    # entirely.  The trainer will read pseudo-labels from disk; any rare cache
    # miss simply won't have a distillation term that step.
    _has_cache = bool(data_cfg.get("teacher_cache_dir"))
    _skip_live = train_cfg.get("skip_live_teachers", _has_cache)

    if not distillation_requested:
        if teacher_cfgs:
            logger.info("Distillation disabled in config; skipping teacher loading.")
        if _has_cache:
            logger.info("Teacher cache present but distillation is disabled; cached teacher depths will be ignored.")
    elif _skip_live and _has_cache:
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
        if distillation_requested and not _has_cache:
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
    distillation_cfg = dict(cfg.get("distillation", {}))
    distillation_cfg["enabled"] = distillation_requested
    if distillation_requested:
        if active_teachers:
            distillation_cfg.setdefault(
                "teachers",
                [
                    {"name": n, "weight": teacher_weights.get(n, 1.0), "loss": "si_log"}
                    for n in active_teachers.keys()
                ],
            )
        elif _has_cache and teacher_cfgs:
            distillation_cfg.setdefault(
                "teachers",
                [
                    {"name": t["name"], "weight": float(t.get("weight", 1.0)), "loss": "si_log"}
                    for t in teacher_cfgs
                    if t.get("enabled", True)
                ],
            )
            logger.info(
                "Distillation enabled from cached teacher depths: %s",
                [t["name"] for t in distillation_cfg["teachers"]],
            )
        else:
            logger.warning(
                "Distillation is enabled but no live teachers or teacher cache are available."
            )
    elif _has_cache and teacher_cfgs:
        logger.info("Ignoring cached teacher depths because distillation.enabled=false.")
    criterion = CombinedLoss(cfg=loss_cfg, distillation_cfg=distillation_cfg)

    optimizer = _build_optimizer(model, train_cfg, logger)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # bf16 has fp32 exponent range; no loss scaling needed

    max_epochs = int(train_cfg.get("max_epochs", 50))
    warmup_steps = train_cfg.get("warmup_steps", 200)
    grad_accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    # n_train_batches is captured from the DataLoader that already has
    # DistributedSampler attached, so it already reflects the per-rank step
    # count (DistributedSampler.__len__ returns ceil(len(dataset)/world_size)).
    # No further division by world_size is needed.
    # Scheduler must be parameterized by optimizer steps (not micro-batches)
    # when gradient accumulation is enabled.
    steps_per_epoch = max(math.ceil(max(n_train_batches, 1) / grad_accum_steps), 1)
    total_steps = max_epochs * steps_per_epoch

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=1e-6,
    )
    logger.info(
        "Scheduler: warmup_steps=%d total_steps=%d steps_per_epoch=%d (grad_accum=%d)",
        warmup_steps,
        total_steps,
        steps_per_epoch,
        grad_accum_steps,
    )
    if warmup_steps >= total_steps:
        logger.warning(
            "warmup_steps (%d) >= total_steps (%d): LR may stay in warmup for most/all training",
            warmup_steps,
            total_steps,
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
    log_dir = artifacts.log_dir
    metrics_logger: JsonlMetricLogger | None = None
    selection_metric = _resolve_selection_metric(train_cfg, loss_cfg)
    if is_main:
        write_run_metadata(
            artifacts=artifacts,
            cfg=cfg,
            args={
                "config": args.config,
                "resume": args.resume,
                "device": args.device,
                "debug": args.debug,
                "verbose": args.verbose,
            },
            project_root=Path.cwd(),
            extra={
                "dataset": dataset_type,
                "selection_metric": selection_metric,
            },
        )
        writer = SummaryWriter(str(log_dir))
        writer.add_text("config", yaml.safe_dump(cfg, sort_keys=False), 0)
        metrics_logger = JsonlMetricLogger(artifacts.metrics_file)
        metrics_logger.log_event(
            "run_start",
            step=0,
            epoch=0,
            dataset=dataset_type,
            selection_metric=selection_metric,
            target_mode=str(loss_cfg.get("training_target", "metric")).lower(),
            trainable_params=n_params,
            model_type=model_type,
            world_size=world_size,
        )
    else:
        writer = None
    logger.info("TensorBoard logs -> %s (run: tensorboard --logdir %s)", log_dir, log_dir)
    if is_main:
        logger.info(
            "Run artifacts -> verbose_log=%s metrics=%s metadata=%s config_snapshot=%s",
            artifacts.verbose_log_file,
            artifacts.metrics_file,
            artifacts.metadata_file,
            artifacts.config_snapshot_file,
        )

    log_img_every = train_cfg.get("log_images_every_n_steps", 50)
    log_img_max_B = train_cfg.get("log_images_max_batch", 2)
    step_log_every = int(train_cfg.get("step_log_every_n_steps", 0))
    if args.debug and step_log_every < 1:
        step_log_every = 1

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", f"checkpoints/{dataset_type}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Checkpoint selection metric: %s (%s is better)",
        selection_metric,
        "higher" if _selection_metric_higher_is_better(selection_metric) else "lower",
    )

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    best_selection_value = _initial_best_selection_value(selection_metric)
    global_step = 0

    if args.resume and os.path.isfile(args.resume):
        resume_model_only = bool(train_cfg.get("resume_model_only", False))
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get("model_is_ema", False):
            logger.warning(
                "Resuming from an EMA-weight checkpoint (%s). "
                "For continued training prefer latest.pt unless this is intentional.",
                args.resume,
            )
        model.load_state_dict(ckpt["model"])

        if resume_model_only:
            logger.info(
                "Resume mode: model-only (optimizer/scheduler/scaler reset, start_epoch=0)."
            )
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
                # Enforce current run schedule horizons even when resuming from an
                # older checkpoint trained with a different total_steps policy.
                scheduler.warmup_steps = warmup_steps
                scheduler.total_steps = total_steps
                scheduler.min_lr = 1e-6
                logger.info(
                    "Scheduler resumed: last_epoch=%d warmup_steps=%d total_steps=%d",
                    scheduler.last_epoch,
                    scheduler.warmup_steps,
                    scheduler.total_steps,
                )
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
            ckpt_selection_metric = _normalize_selection_metric(
                ckpt.get("best_selection_metric", selection_metric)
            )
            if ckpt_selection_metric == selection_metric and "best_selection_value" in ckpt:
                best_selection_value = float(ckpt["best_selection_value"])
            elif selection_metric == "loss":
                best_selection_value = best_val_loss
            else:
                logger.warning(
                    "Checkpoint does not contain compatible best-selection metadata for '%s'; "
                    "selection tracking will restart for this run.",
                    selection_metric,
                )
            global_step = ckpt.get("global_step", 0)
            if ema is not None and "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            logger.info(
                "Resumed from epoch %d, best_val_loss=%.5f, best_%s=%.5f",
                start_epoch,
                best_val_loss,
                selection_metric,
                best_selection_value,
            )
    elif args.resume:
        logger.warning("Resume checkpoint not found: %s", args.resume)

    # Optional: run N additional epochs after resume (stage transition mode).
    resume_additional_epochs = int(train_cfg.get("resume_additional_epochs", 0))
    if args.resume and resume_additional_epochs > 0:
        max_epochs = start_epoch + resume_additional_epochs
        total_steps = max_epochs * steps_per_epoch
        scheduler.warmup_steps = warmup_steps
        scheduler.total_steps = total_steps
        scheduler.min_lr = 1e-6
        curriculum.total_epochs = max_epochs
        logger.info(
            "resume_additional_epochs=%d -> max_epochs set to %d (start_epoch=%d, total_steps=%d)",
            resume_additional_epochs,
            max_epochs,
            start_epoch,
            total_steps,
        )

    if start_epoch >= max_epochs:
        raise RuntimeError(
            f"No epochs left to train: start_epoch={start_epoch} max_epochs={max_epochs}. "
            "Increase training.max_epochs, or set training.resume_additional_epochs > 0, "
            "or set training.resume_model_only=true for stage transitions."
        )

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
    unfreeze_backbone_epoch = train_cfg.get("unfreeze_backbone_epoch")
    backbone_unfrozen = not bool(train_cfg.get("freeze_backbone", False))

    for epoch in epoch_bar:
        epoch_t0 = time.time()

        if (
            unfreeze_backbone_epoch is not None
            and not backbone_unfrozen
            and (epoch + 1) >= int(unfreeze_backbone_epoch)
        ):
            _unfreeze_backbone_group(model, optimizer, scheduler, train_cfg, logger)
            backbone_unfrozen = True

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
            grad_accum_steps=grad_accum_steps,
            logger=logger,
            step_log_every_n_steps=step_log_every,
            is_main=is_main,
        )
        logger.info("Epoch %d finished | train_loss=%.6f | global_step=%d", epoch + 1, avg_loss, global_step)
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
            if metrics_logger is not None:
                metrics_logger.log_event(
                    "train_epoch",
                    step=global_step,
                    epoch=epoch + 1,
                    metrics={
                        "loss": avg_loss,
                        "degradation_scale": deg_scale,
                        "epoch_time_s": epoch_time,
                        "eta_min": eta_m,
                    },
                )

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
            selection_value = _extract_selection_value(val_metrics, selection_metric)
            saved_best = False
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
                logger.info(
                    "Validation epoch %d | loss=%.5f abs_rel=%.5f delta1=%.4f rmse=%.4f si_log=%.5f",
                    epoch + 1,
                    val_loss,
                    abs_rel,
                    delta1,
                    val_metrics.get("rmse", 0),
                    val_metrics.get("si_log", 0),
                )

            # Save best checkpoint (rank 0 only)
            if is_main and _is_better_selection_value(selection_metric, selection_value, best_selection_value):
                best_val_loss = val_loss
                best_selection_value = selection_value
                saved_best = True
                if ema is not None:
                    ema.apply_shadow()
                    try:
                        _save_ckpt(
                            ckpt_dir / "best.pt",
                            epoch, model, optimizer, scaler, scheduler,
                            global_step, best_val_loss, selection_metric, best_selection_value,
                            cfg, ema, model_is_ema=True,
                        )
                    finally:
                        ema.restore()
                else:
                    _save_ckpt(
                        ckpt_dir / "best.pt",
                        epoch, model, optimizer, scaler, scheduler,
                        global_step, best_val_loss, selection_metric, best_selection_value,
                        cfg, ema, model_is_ema=False,
                    )
                tqdm.write(f"  -> Best saved ({selection_metric}={best_selection_value:.5f})")
                logger.info(
                    "New best checkpoint saved: %s=%.5f (epoch=%d, val_loss=%.5f)",
                    selection_metric,
                    best_selection_value,
                    epoch + 1,
                    val_loss,
                )
            if is_main and metrics_logger is not None:
                metrics_logger.log_event(
                    "validation",
                    step=global_step,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    selection_metric=selection_metric,
                    selection_value=selection_value,
                    checkpoint_saved=saved_best,
                )

        # Save latest checkpoint every epoch (rank 0 only)
        if is_main:
            _save_ckpt(
                ckpt_dir / "latest.pt",
                epoch, model, optimizer, scaler, scheduler,
                global_step, best_val_loss, selection_metric, best_selection_value,
                cfg, ema, model_is_ema=False,
            )
        logger.info("Saved latest checkpoint for epoch %d", epoch + 1)

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
            if metrics_logger is not None:
                metrics_logger.log_event("latency", metrics=latency)
            if not latency["meets_target"]:
                logger.warning("Does not meet 30 FPS target")
        except Exception as e:
            logger.warning("Latency profiling skipped: %s", e)

        if metrics_logger is not None:
            metrics_logger.log_event(
                "run_complete",
                step=global_step,
                epoch=max_epochs,
                best_val_loss=best_val_loss,
                best_selection_metric=selection_metric,
                best_selection_value=best_selection_value,
            )
        writer.close()
        if metrics_logger is not None:
            metrics_logger.close()

    logger.info(
        "Training complete. Best %s: %.5f | Best val loss: %.5f",
        selection_metric,
        best_selection_value,
        best_val_loss,
    )
    logger.info("Checkpoints in: %s", ckpt_dir)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
