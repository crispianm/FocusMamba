#!/usr/bin/env python3
"""
test_training.py — Trial Training Script (YouTube-VOS, L1 distillation)
=========================================================================

A minimal, self-contained training loop for iterative development.
Does NOT depend on the training/ package stubs — uses plain L1 loss only.

Trains the FocusMamba student on clean YouTube-VOS clips, distilling
knowledge from frozen teacher models (VideoDepthAnything, DepthPro, DA3).

Features:
    - L1 loss between student and per-teacher pseudo-GT depth
    - TensorBoard: scalar losses + RGB / depth image grids every N steps
    - Best / latest checkpoint saving
    - --resume support
    - Clean teacher failure handling (missing checkpoint → warning, skip)

Usage:
    python test_training.py --config configs/experiments/trial_youtube_vos.yaml
    python test_training.py --config configs/experiments/trial_youtube_vos.yaml \
        --resume checkpoints/trial_youtube_vos/latest.pt
    python test_training.py --config configs/experiments/trial_youtube_vos.yaml \
        --device cpu  # for quick local sanity check
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import yaml

from models import build_model
from models.teachers import build_teacher
from dataloader.youtube_vos import YouTubeVOSDataset


# ---------------------------------------------------------------------------
# Depth visualisation helper
# ---------------------------------------------------------------------------

def depth_to_rgb(depth: torch.Tensor) -> torch.Tensor:
    """Normalise a (H, W) depth map to a (3, H, W) uint8-range float tensor.

    Uses a turbo-like colourmap approximation that requires no matplotlib at
    training time.  The colourmap is hard-coded as a 256-entry lookup table
    derived from the matplotlib 'turbo' colourmap (Apache 2.0 licence).
    """
    # Normalise to [0, 1]
    d_min = depth.min()
    d_max = depth.max()
    if d_max - d_min > 1e-6:
        norm = (depth - d_min) / (d_max - d_min)
    else:
        norm = torch.zeros_like(depth)

    # Approximate turbo colourmap via a polynomial (matches matplotlib turbo)
    r = torch.clamp(
        0.1357 + norm * (4.5974 + norm * (-12.494 + norm * (8.9094 + norm * (-1.2689)))),
        0.0, 1.0,
    )
    g = torch.clamp(
        0.09133 + norm * (2.5906 + norm * (0.23823 + norm * (-5.4571 + norm * 3.3436))),
        0.0, 1.0,
    )
    b = torch.clamp(
        0.1597 + norm * (5.0823 + norm * (-14.289 + norm * (13.489 + norm * (-4.4436)))),
        0.0, 1.0,
    )
    return torch.stack([r, g, b], dim=0)  # (3, H, W)


def make_image_grid(
    frames: torch.Tensor,           # (B, 3, T, H, W)
    student_depth: torch.Tensor,    # (B, 1, T, H, W)
    teacher_depths: Dict[str, torch.Tensor],  # name → (B, 1, T, H, W)
    max_samples: int = 2,
    frame_idx: int = 0,
) -> torch.Tensor:
    """
    Assemble an (3, H, N*W) image strip for TensorBoard.

    Each column is one sample from the batch; each row is:
      [RGB input | student depth | teacher_1 depth | teacher_2 depth | …]
    """
    B = min(frames.shape[0], max_samples)
    rows = []

    for b in range(B):
        rgb = frames[b, :, frame_idx].cpu()           # (3, H, W) in [0,1]
        s_d = student_depth[b, 0, frame_idx].cpu()    # (H, W)
        col = [rgb, depth_to_rgb(s_d)]
        for t_depth in teacher_depths.values():
            t_d = t_depth[b, 0, frame_idx].cpu()      # (H, W)
            col.append(depth_to_rgb(t_d))
        # stack column vertically
        rows.append(torch.cat(col, dim=1))  # (3, sum_of_H, W)

    # concatenate columns horizontally
    grid = torch.cat(rows, dim=2)  # (3, total_H, total_W)
    return grid.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distillation_l1(
    pred: torch.Tensor,              # (B, 1, T, H, W)
    teacher_depths: Dict[str, torch.Tensor],
    teacher_weights: Dict[str, float],
) -> torch.Tensor:
    """Weighted average L1 loss across all teachers.

    L1 expects predictions and pseudo-GT to be at the same scale.
    Teachers produce metric depth; student produces log-metric depth via exp,
    so both should be in metres — as long as the model's decoder uses exp().
    """
    total_weight = sum(teacher_weights.values()) or 1.0
    loss = pred.new_zeros(1)
    B, C, T, H, W = pred.shape
    for name, t_depth in teacher_depths.items():
        w = teacher_weights.get(name, 1.0)
        _B, _C, _T, _Ht, _Wt = t_depth.shape
        if (_Ht, _Wt) != (H, W):
            t_depth = t_depth.reshape(B * T, 1, _Ht, _Wt)
            t_depth = F.interpolate(t_depth, size=(H, W), mode="bilinear", align_corners=False)
            t_depth = t_depth.reshape(B, 1, T, H, W)
        loss = loss + w * F.l1_loss(pred, t_depth)
    return loss / total_weight


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FocusMamba trial training run")
    parser.add_argument("--config", default="configs/experiments/trial_youtube_vos.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    train_cfg = cfg.get("training", {})
    use_amp = train_cfg.get("precision", "bf16") == "bf16" and device.type == "cuda"

    # -----------------------------------------------------------------------
    # Student model
    # -----------------------------------------------------------------------
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Student parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # Teacher models — loaded lazily on first predict(); failures are caught
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

    if not active_teachers:
        print("WARNING: No teachers registered. Training with zero loss — check configs.")

    # -----------------------------------------------------------------------
    # Dataset and dataloaders
    # -----------------------------------------------------------------------
    data_cfg = cfg.get("data", {})
    full_dataset = YouTubeVOSDataset(
        root=data_cfg.get("root", "data/youtube-vos/train_all_frames/JPEGImages"),
        num_frames=data_cfg.get("num_frames", 8),
        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        max_videos=data_cfg.get("max_videos", 100),
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
        max_videos=data_cfg.get("max_videos", 100),
        clip_stride=data_cfg.get("clip_stride", 8),
        frame_stride=data_cfg.get("frame_stride", 1),
        split="val",
        val_fraction=data_cfg.get("val_fraction", 0.1),
        seed=data_cfg.get("seed", 42),
    )

    batch_size = train_cfg.get("batch_size", 2)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        full_dataset,
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
    print(f"Train clips: {len(full_dataset)}  |  Val clips: {len(val_dataset)}")

    # -----------------------------------------------------------------------
    # Optimiser, scaler, scheduler
    # -----------------------------------------------------------------------
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

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -----------------------------------------------------------------------
    # TensorBoard
    # -----------------------------------------------------------------------
    log_dir = Path(train_cfg.get("log_dir", "runs/trial_youtube_vos"))
    writer = SummaryWriter(str(log_dir))
    writer.add_text("config", yaml.dump(cfg), 0)
    print(f"TensorBoard logs → {log_dir}  (run: tensorboard --logdir {log_dir})")

    log_img_every = train_cfg.get("log_images_every_n_steps", 50)
    log_img_max_B = train_cfg.get("log_images_max_batch", 2)

    # -----------------------------------------------------------------------
    # Checkpoint dir
    # -----------------------------------------------------------------------
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/trial_youtube_vos"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch},  best_val_loss={best_val_loss:.5f}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    grad_clip = train_cfg.get("grad_clip", 1.0)
    val_every = train_cfg.get("val_every_n_epochs", 5)
    t0 = time.time()

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_losses: List[float] = []
        epoch_t0 = time.time()

        for batch in train_loader:
            frames = batch["frames"].to(device)  # (B, 3, T, H, W)

            # ── Teacher inference (frozen, no grad) ─────────────────────────
            pseudo_depths: Dict[str, torch.Tensor] = {}
            for t_name, teacher in active_teachers.items():
                try:
                    with torch.no_grad():
                        pd = teacher.predict(frames)  # (B, 1, T, H, W)
                    pseudo_depths[t_name] = pd.detach()
                except NotImplementedError:
                    # Teacher not yet implemented — skip silently
                    pass
                except Exception as e:
                    print(f"  Teacher '{t_name}' error (step {global_step}): {e}")

            if not pseudo_depths:
                # No teachers available yet — skip this batch
                scheduler.step()
                global_step += 1
                continue

            # ── Student forward + loss ───────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(frames)
                pred_depth = outputs["depth"]  # (B, 1, T, H, W)
                loss = distillation_l1(pred_depth, pseudo_depths, teacher_weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            writer.add_scalar("train/l1_loss", loss_val, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            # ── TensorBoard image grid ───────────────────────────────────────
            if global_step % log_img_every == 0 and pseudo_depths:
                with torch.no_grad():
                    grid = make_image_grid(
                        frames, pred_depth.detach(), pseudo_depths,
                        max_samples=log_img_max_B,
                    )
                writer.add_image("train/depth_grid", grid, global_step)
                # Also log per-teacher depth scale for monitoring
                for t_name, pd in pseudo_depths.items():
                    writer.add_scalar(
                        f"teacher/{t_name}_mean_depth",
                        pd.mean().item(),
                        global_step,
                    )
                writer.add_scalar(
                    "student/mean_depth", pred_depth.mean().item(), global_step
                )

            global_step += 1

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        elapsed = time.time() - t0
        epochs_done = epoch - start_epoch + 1
        eta_min = elapsed / epochs_done * (max_epochs - epoch - 1) / 60.0
        epoch_secs = time.time() - epoch_t0

        print(
            f"Epoch {epoch + 1}/{max_epochs}  "
            f"train_l1={avg_loss:.5f}  "
            f"time={epoch_secs:.1f}s  "
            f"ETA={eta_min:.1f}min"
        )
        writer.add_scalar("train/epoch_l1", avg_loss, epoch)

        # ── Validation ──────────────────────────────────────────────────────
        if (epoch + 1) % val_every == 0 or epoch == max_epochs - 1:
            model.eval()
            val_losses: List[float] = []
            val_pseudo: Optional[Dict[str, torch.Tensor]] = None  # for image log

            with torch.no_grad():
                for batch in val_loader:
                    frames = batch["frames"].to(device)
                    v_pseudo: Dict[str, torch.Tensor] = {}
                    for t_name, teacher in active_teachers.items():
                        try:
                            pd = teacher.predict(frames)
                            v_pseudo[t_name] = pd.detach()
                        except Exception:
                            pass

                    if not v_pseudo:
                        continue

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(frames)
                        pred = outputs["depth"]
                        vloss = distillation_l1(pred, v_pseudo, teacher_weights)
                    val_losses.append(vloss.item())

                    # save one batch for image logging
                    if val_pseudo is None:
                        val_pseudo = {k: v[:log_img_max_B] for k, v in v_pseudo.items()}
                        val_pred = pred[:log_img_max_B].detach()
                        val_frames_log = frames[:log_img_max_B]

            avg_val = sum(val_losses) / max(len(val_losses), 1)
            print(f"  Val  l1={avg_val:.5f}")
            writer.add_scalar("val/epoch_l1", avg_val, epoch)

            if val_pseudo is not None:
                grid = make_image_grid(val_frames_log, val_pred, val_pseudo, max_samples=log_img_max_B)
                writer.add_image("val/depth_grid", grid, epoch)

            # Save best checkpoint
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                _save_ckpt(
                    ckpt_dir / "best.pt",
                    epoch, model, optimizer, scaler, scheduler, global_step, best_val_loss, cfg,
                )
                print(f"  → Best saved (val_l1={best_val_loss:.5f})")

        # Save latest checkpoint every epoch
        _save_ckpt(
            ckpt_dir / "latest.pt",
            epoch, model, optimizer, scaler, scheduler, global_step, best_val_loss, cfg,
        )

    writer.close()
    print(f"\nDone. Best val L1: {best_val_loss:.5f}")
    print(f"Checkpoints in: {ckpt_dir}")


# ---------------------------------------------------------------------------
# Helpers
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
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "config": cfg,
        },
        path,
    )


if __name__ == "__main__":
    main()
