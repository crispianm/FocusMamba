#!/usr/bin/env python3
"""
FocusMamba Training Script
===========================

Features:
    - bf16 mixed precision with GradScaler.
    - Gradient clipping.
    - Linear warmup → cosine decay LR schedule.
    - TensorBoard logging of all losses and metrics.
    - Best-checkpoint saving based on val MAE.
    - Optional --baseline flag for a 2-D UNet ablation.
    - ETA logging.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --baseline
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import yaml
from tqdm import tqdm

from dataloader.focus_dataset import build_dataloaders
from dataloader.degradation import build_degradation
from models import build_model, FocusMamba
from models.focus_transformer import FocusTransformer
from utils.loss import FocusLoss
from utils.metrics import MetricSuite


# ---------------------------------------------------------------------------
# Simple 2-D UNet Baseline (per-frame, for ablation)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BaselineUNet2D(nn.Module):
    """Minimal per-frame 2-D UNet for ablation comparison."""

    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, base_ch)
        self.enc2 = _DoubleConv(base_ch, base_ch * 2)
        self.enc3 = _DoubleConv(base_ch * 2, base_ch * 4)
        self.bottleneck = _DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = _DoubleConv(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Process a single frame (B, 3, H, W) → (B, 1, H, W)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.head(d1))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) → (B, 1, T, H, W) by processing each frame independently."""
        B, C, T, H, W = frames.shape
        out = []
        for t in range(T):
            out.append(self.forward_frame(frames[:, :, t]))
        return torch.stack(out, dim=2)  # (B, 1, T, H, W)


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
# Training & Validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: FocusLoss,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[float, float, int, int, int]:
    model.train()
    running_loss = 0.0
    running_l1 = 0.0

    # Progress bar showing iterations/sec; disappears after epoch completes.
    for batch in tqdm(loader, desc="Train", unit="it", leave=False, total=len(loader)):
        frames = batch["frames"].to(device)          # (B,C,T,H,W)
        focus_gt = batch["focus_maps"].to(device)     # (B,1,T,H,W)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            pred = model(frames)
            # Guard: skip this batch if model produced non-finite predictions
            # (can happen when SSM state recurrence overflows for an unlucky batch).
            if not torch.isfinite(pred).all():
                optimizer.zero_grad(set_to_none=True)
                continue
            losses = criterion(pred, focus_gt)

        if not torch.isfinite(losses):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)

        # Guard: if the backward pass through selective_scan_cuda produced
        # non-finite gradients, skip this update entirely rather than letting
        # NaN weights corrupt all future forward passes.
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

        running_loss += losses.item()
        running_l1 += losses.item()
        global_step += 1

        # TensorBoard logging
        if global_step % 20 == 0:
            writer.add_scalar("train/loss_total", losses.item(), global_step)
            # writer.add_scalar("train/loss_l1", losses["l1"].item(), global_step)
            # writer.add_scalar("train/loss_ssim", losses["ssim"].item(), global_step)
            # writer.add_scalar("train/loss_grad", losses["grad"].item(), global_step)
            # writer.add_scalar("train/loss_tgm", losses["tgm"].item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    avg_loss = running_loss / len(loader)

    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: FocusLoss,
    metrics_fn: MetricSuite,
    device: torch.device,
    use_amp: bool,
    writer: SummaryWriter,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    metric_accum: Dict[str, float] = {}
    n_batches = 0

    # Progress bar for validation that also disappears after completion.
    for batch in tqdm(loader, desc=f"Val {epoch}", unit="it", leave=False, total=len(loader)):
        frames = batch["frames"].to(device)
        focus_gt = batch["focus_maps"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            pred = model(frames)
            losses = criterion(pred, focus_gt)

        if not torch.isfinite(losses):
            continue

        total_loss += losses.item()

        # Metrics
        m = metrics_fn(pred.float(), focus_gt.float())
        for k, v in m.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v.item()
        n_batches += 1

    if n_batches == 0:
        avg_loss = float("inf")
        avg_metrics = {
            "mae": float("inf"),
            "ssim": 0.0,
            "psnr": 0.0,
            "weighted_f": 0.0,
            "tss": float("inf"),
        }
    else:
        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in metric_accum.items()}

    # Log to TensorBoard
    writer.add_scalar("val/loss_total", avg_loss, epoch)
    for k, v in avg_metrics.items():
        writer.add_scalar(f"val/{k}", v, epoch)

    avg_metrics["loss"] = avg_loss
    return avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train FocusMamba")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--baseline", action="store_true", help="Use 2D UNet baseline instead")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: bfloat16 AMP causes the Mamba selective_scan_cuda kernel to produce
    # NaN because the exp(delta*A) recurrence overflows in bf16's 7-bit mantissa.
    # The default config ships with precision=fp32.  Only enable bf16 if you have
    # verified your selective_scan_cuda build is numerically stable at that dtype.
    use_amp = cfg.get("precision", "bf16") == "bf16" and device.type == "cuda"

    # Build degradation pipeline (if configured)
    lowlight_deg = build_degradation(cfg)
    if lowlight_deg is not None:
        deg_cfg = cfg.get("degradation", {})
        print(f"Low-light degradation ENABLED  (lux={deg_cfg.get('lux_level', 10.0)})")
    else:
        print("Low-light degradation disabled")

    # Dataloaders
    loaders = build_dataloaders(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        window_length=cfg["window_length"],
        stride=cfg.get("clip_stride", 4),
        crop_size=tuple(cfg["resolution"]),
        max_coc=cfg["max_coc"],
        focal_length_range=tuple(cfg["focal_length_range"]),
        aperture_range=tuple(cfg["aperture_range"]),
        beta_alpha=cfg["beta_alpha"],
        beta_beta=cfg["beta_beta"],
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        split_seed=cfg.get("split_seed", 42),
        lowlight_degradation=lowlight_deg,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Model
    model_type = cfg.get("model_type", "mamba")
    if args.baseline:
        model = BaselineUNet2D(in_channels=3).to(device)
        model_tag = "baseline_unet2d"
    else:
        model = build_model(cfg).to(device)
        model_tag = f"focus_{model_type}"

    # Build descriptive noise tag for logging
    deg_cfg = cfg.get("degradation", {})
    noise_tag = f"lux{deg_cfg.get('lux_level', 'off')}" if deg_cfg.get("enabled", False) else "clean"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_tag}  |  Parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    # criterion = FocusLoss(
    #     lambda_l1=cfg["lambda_l1"],
    #     lambda_ssim=cfg["lambda_ssim"],
    #     lambda_grad=cfg["lambda_grad"],
    #     lambda_tgm=cfg["lambda_tgm"],
    # )
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = cfg["max_epochs"] * max(len(train_loader), 1)
    scheduler = WarmupCosineScheduler(
        optimizer, cfg["warmup_steps"], total_steps, min_lr=1e-6,
    )

    metrics_fn = MetricSuite(compute_lpips=False)

    # Logging — tag runs with model type and noise level
    log_dir = Path(cfg.get("log_dir", "./runs")) / f"{model_tag}_{noise_tag}"
    writer = SummaryWriter(str(log_dir))
    # Log config metadata for reproducibility
    writer.add_text("config/model_type", model_type if not args.baseline else "baseline", 0)
    writer.add_text("config/noise_level", noise_tag, 0)
    ckpt_dir = Path(cfg.get("checkpoint_dir", "./checkpoints")) / model_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 0
    best_mae = float("inf")
    global_step = 0
    resumed_from = None
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", float("inf"))
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}, best_mae={best_mae:.5f}")

    # Training loop
    val_every = cfg.get("val_every_n_epochs", 2)
    t_start = time.time()

    for epoch in range(start_epoch, cfg["max_epochs"]):
        epoch_t0 = time.time()
        avg_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, cfg["grad_clip"], use_amp, writer, global_step,
        )
        epoch_time = time.time() - epoch_t0

        # ETA
        elapsed = time.time() - t_start
        epochs_done = epoch - start_epoch + 1
        eta_s = elapsed / epochs_done * (cfg["max_epochs"] - epoch - 1)
        eta_m = eta_s / 60.0

        print(
            f"Epoch {epoch+1}/{cfg['max_epochs']}  "
            f"train_loss={avg_loss:.4f}  "
            f"time={epoch_time:.1f}s  "
            f"ETA={eta_m:.1f}min"
        )
        writer.add_scalar("train/epoch_loss_total", avg_loss, epoch)

        # Validation
        if (epoch + 1) % val_every == 0 or epoch == cfg["max_epochs"] - 1:
            val_metrics = validate(
                model, val_loader, criterion, metrics_fn, device, use_amp, writer, epoch,
            )
            print(
                f"  Val — loss={val_metrics['loss']:.4f}  "
                f"MAE={val_metrics['mae']:.5f}  "
                f"SSIM={val_metrics.get('ssim', 0):.4f}  "
                f"PSNR={val_metrics.get('psnr', 0):.2f}  "
                f"Fw={val_metrics.get('weighted_f', 0):.4f}  "
                f"TSS={val_metrics.get('tss', 0):.5f}"
            )

            # Save best checkpoint (based on val MAE)
            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_mae": best_mae,
                        "global_step": global_step,
                        "config": cfg,
                    },
                    ckpt_dir / "best.pt",
                )
                print(f"  ✓ Best checkpoint saved (MAE={best_mae:.5f})")

        # Save latest checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_mae": best_mae,
                "global_step": global_step,
                "config": cfg,
            },
            ckpt_dir / "latest.pt",
        )

    writer.close()
    print(f"Training complete. Best val MAE: {best_mae:.5f}")


if __name__ == "__main__":
    main()
