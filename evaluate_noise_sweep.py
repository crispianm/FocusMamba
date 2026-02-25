#!/usr/bin/env python3
"""
Noise Sweep Evaluator
======================

Takes two pre-trained checkpoints (one Mamba, one Transformer) and evaluates
both over the test set at progressively increasing noise levels.

Outputs:
    - Console table of MAE at each noise level.
    - A matplotlib chart saved to ``noise_sweep_results.png``.
    - A CSV log saved to ``noise_sweep_results.csv``.

Usage:
    python evaluate_noise_sweep.py \\
        --mamba_ckpt  checkpoints/focus_mamba/best.pt \\
        --transformer_ckpt checkpoints/focus_transformer/best.pt \\
        --config config.yaml \\
        --lux_levels 50 30 20 10 5 3 1 \\
        --output_dir ./eval_results
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader.focus_dataset import build_dataloaders
from dataloader.degradation import LowLightDegradation
from models import build_model
from utils.metrics import MetricSuite, noise_robustness_score


# ---------------------------------------------------------------------------
# Evaluation at a single noise level
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_at_noise_level(
    model: torch.nn.Module,
    loader,
    metrics_fn: MetricSuite,
    device: torch.device,
    use_amp: bool,
    degradation: LowLightDegradation | None = None,
) -> Dict[str, float]:
    """Run evaluation on the entire loader, optionally applying degradation.

    If *degradation* is provided, it is applied to *frames* on-the-fly before
    feeding them to the model.  GT focus maps remain clean.
    """
    model.eval()
    metric_accum: Dict[str, float] = {}
    n_batches = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        frames = batch["frames"].to(device)        # (B, C, T, H, W)
        focus_gt = batch["focus_maps"].to(device)   # (B, 1, T, H, W)

        # Apply degradation at inference time (if testing noise robustness)
        if degradation is not None:
            B, C, T, H, W = frames.shape
            degraded_frames = []
            rng = np.random.RandomState(n_batches)  # deterministic per batch
            for b in range(B):
                # (C, T, H, W) → (T, C, H, W)
                f = frames[b].permute(1, 0, 2, 3).contiguous()
                f = degradation(f, rng=rng)
                degraded_frames.append(f.permute(1, 0, 2, 3))  # back to (C, T, H, W)
            frames = torch.stack(degraded_frames, dim=0)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            pred = model(frames)

        m = metrics_fn(pred.float(), focus_gt.float())
        for k, v in m.items():
            metric_accum[k] = metric_accum.get(k, 0.0) + v.item()
        n_batches += 1

    if n_batches == 0:
        return {"mae": float("inf"), "ssim": 0.0, "psnr": 0.0}

    return {k: v / n_batches for k, v in metric_accum.items()}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Noise sweep evaluation")
    parser.add_argument("--mamba_ckpt", type=str, required=True, help="Path to Mamba checkpoint")
    parser.add_argument("--transformer_ckpt", type=str, required=True, help="Path to Transformer checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--lux_levels", type=float, nargs="+",
        default=[50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 1.0],
        help="Lux levels to sweep (descending = increasing noise)",
    )
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("precision", "bf16") == "bf16" and device.type == "cuda"

    # Build test loader (without degradation — we apply it per-sweep-step)
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
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
        lowlight_degradation=None,  # no degradation in the dataloader
    )
    test_loader = loaders.get("test", loaders.get("val"))
    if test_loader is None or len(test_loader) == 0:
        print("ERROR: No test or val data found.")
        sys.exit(1)

    metrics_fn = MetricSuite(compute_lpips=False)

    # ---------- Load models ----------
    print("Loading Mamba checkpoint:", args.mamba_ckpt)
    cfg_mamba = dict(cfg)
    cfg_mamba["model_type"] = "mamba"
    mamba_model = build_model(cfg_mamba).to(device)
    mamba_ckpt = torch.load(args.mamba_ckpt, map_location=device)
    mamba_model.load_state_dict(mamba_ckpt["model"])
    mamba_model.eval()

    print("Loading Transformer checkpoint:", args.transformer_ckpt)
    cfg_trans = dict(cfg)
    cfg_trans["model_type"] = "transformer"
    transformer_model = build_model(cfg_trans).to(device)
    trans_ckpt = torch.load(args.transformer_ckpt, map_location=device)
    transformer_model.load_state_dict(trans_ckpt["model"])
    transformer_model.eval()

    # ---------- Noise sweep ----------
    lux_levels = sorted(args.lux_levels, reverse=True)  # high lux (clean) → low lux (noisy)
    deg_cfg = cfg.get("degradation", {})

    results: List[Dict] = []
    mamba_clean_mae = None
    trans_clean_mae = None

    print("\n" + "=" * 72)
    print(f"{'Lux':>8s}  {'Mamba MAE':>12s}  {'Trans MAE':>12s}  {'Mamba NRS':>10s}  {'Trans NRS':>10s}")
    print("=" * 72)

    for lux in lux_levels:
        # Build degradation for this lux level
        if lux >= 50.0:
            degradation = None  # clean evaluation at lux=50
        else:
            degradation = LowLightDegradation(
                lux_level=lux,
                poisson_scale=deg_cfg.get("poisson_scale", 0.02),
                gaussian_std=deg_cfg.get("gaussian_std", 0.01),
                blur_kernel_size=deg_cfg.get("blur_kernel_size", 15),
                blur_intensity=deg_cfg.get("blur_intensity", 1.0) * (50.0 / max(lux, 1.0)) ** 0.3,
                temporal_variance=deg_cfg.get("temporal_variance", 0.05),
            )

        # Evaluate both models
        mamba_metrics = evaluate_at_noise_level(
            mamba_model, test_loader, metrics_fn, device, use_amp, degradation,
        )
        trans_metrics = evaluate_at_noise_level(
            transformer_model, test_loader, metrics_fn, device, use_amp, degradation,
        )

        m_mae = mamba_metrics["mae"]
        t_mae = trans_metrics["mae"]

        if mamba_clean_mae is None:
            mamba_clean_mae = m_mae
            trans_clean_mae = t_mae

        m_nrs = noise_robustness_score(mamba_clean_mae, m_mae)
        t_nrs = noise_robustness_score(trans_clean_mae, t_mae)

        row = {
            "lux": lux,
            "mamba_mae": m_mae,
            "transformer_mae": t_mae,
            "mamba_nrs": m_nrs,
            "transformer_nrs": t_nrs,
            "mamba_ssim": mamba_metrics.get("ssim", 0),
            "transformer_ssim": trans_metrics.get("ssim", 0),
            "mamba_psnr": mamba_metrics.get("psnr", 0),
            "transformer_psnr": trans_metrics.get("psnr", 0),
        }
        results.append(row)

        print(
            f"{lux:8.1f}  {m_mae:12.5f}  {t_mae:12.5f}  {m_nrs:10.3f}  {t_nrs:10.3f}"
        )

    print("=" * 72)

    # ---------- Save results ----------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "noise_sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    # Matplotlib chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lux_vals = [r["lux"] for r in results]
        mamba_maes = [r["mamba_mae"] for r in results]
        trans_maes = [r["transformer_mae"] for r in results]

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(lux_vals, mamba_maes, "o-", color="#2196F3", linewidth=2, label="FocusMamba (causal)")
        ax.plot(lux_vals, trans_maes, "s--", color="#F44336", linewidth=2, label="FocusTransformer (bidir)")

        ax.set_xlabel("Illuminance (Lux) — decreasing →", fontsize=12)
        ax.set_ylabel("MAE ↓", fontsize=12)
        ax.set_title("Noise Robustness: Mamba vs Transformer", fontsize=14)
        ax.legend(fontsize=11)
        ax.invert_xaxis()  # lower lux = harder = right side
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        chart_path = output_dir / "noise_sweep_results.png"
        fig.savefig(str(chart_path), dpi=150)
        plt.close(fig)
        print(f"Chart saved to {chart_path}")
    except ImportError:
        print("matplotlib not available; skipping chart generation.")

    print("\nDone.")


if __name__ == "__main__":
    main()
