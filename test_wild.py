#!/usr/bin/env python3
"""
Wild / Zero-Shot Real-World Evaluation
========================================

Loads a trained model (trained purely on degraded Herrmann data) and runs
inference on the WildDataset (uncalibrated smartphone focal sweeps) to
measure zero-shot real-world performance.

Since wild scenes may not have pixel-level focus-map ground truth, this
script supports two modes:

1. **Qualitative**: Save predicted focus maps as images / video for visual
   inspection.
2. **Quantitative** (if labels.json exists): Compute proxy metrics from
   focus-distance labels (e.g., peak sharpness alignment with ground-truth
   focus distance curve).

Usage:
    python test_wild.py \\
        --checkpoint checkpoints/focus_mamba/best.pt \\
        --wild_data ./wild_data \\
        --config config.yaml \\
        --output_dir ./wild_results
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader.wild_dataset import build_wild_dataloader
from models import build_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
) -> List[Dict]:
    """Run model inference on the wild dataset.

    Returns a list of result dicts, one per clip:
        {
            "scene": str,
            "start_frame": int,
            "pred_focus_map": np.ndarray (T, H, W) in [0,1],
            "focus_distances": list[float] | None,
        }
    """
    model.eval()
    results = []

    for batch in tqdm(loader, desc="Wild inference"):
        frames = batch["frames"].to(device)  # (B, C, T, H, W)
        metadata = batch["metadata"]

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            pred = model(frames)  # (B, 1, T, H, W)

        pred_np = pred.cpu().float().numpy()  # (B, 1, T, H, W)

        for b in range(pred_np.shape[0]):
            results.append({
                "scene": metadata[b]["scene"],
                "start_frame": metadata[b]["start_frame"],
                "pred_focus_map": pred_np[b, 0],  # (T, H, W)
                "focus_distances": metadata[b].get("focus_distances"),
            })

    return results


# ---------------------------------------------------------------------------
# Proxy metrics (when focus distances are available)
# ---------------------------------------------------------------------------

def compute_proxy_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute proxy metrics from focus-distance labels.

    For each clip, we compute:
      - **Peak Alignment**: Does the predicted sharpest frame match the GT
        closest-focus frame?  Measured as |argmax(pred_mean) - argmax(GT_focus)|
        normalised by sequence length.
      - **Temporal Smoothness**: Mean absolute frame-to-frame difference in
        predicted spatial-mean focus.
    """
    peak_errors = []
    smoothness_scores = []

    for r in results:
        focus_dists = r.get("focus_distances")
        if focus_dists is None or any(d is None for d in focus_dists):
            continue

        pred_map = r["pred_focus_map"]  # (T, H, W)
        T = pred_map.shape[0]

        # Spatial mean per frame → 1-D sharpness curve
        pred_curve = pred_map.mean(axis=(1, 2))  # (T,)

        # GT: the frame with the smallest focus distance should be "most in focus"
        # (closest focal plane → sharpest)
        gt_dists = np.array(focus_dists[:T], dtype=np.float64)

        pred_peak = int(np.argmax(pred_curve))
        gt_peak = int(np.argmin(gt_dists))  # closest focus = sharpest

        peak_error = abs(pred_peak - gt_peak) / max(T, 1)
        peak_errors.append(peak_error)

        # Temporal smoothness
        diffs = np.abs(np.diff(pred_curve))
        smoothness_scores.append(float(diffs.mean()))

    metrics: Dict[str, float] = {}
    if peak_errors:
        metrics["peak_alignment_error"] = float(np.mean(peak_errors))
        metrics["peak_alignment_std"] = float(np.std(peak_errors))
    if smoothness_scores:
        metrics["temporal_smoothness"] = float(np.mean(smoothness_scores))

    return metrics


# ---------------------------------------------------------------------------
# Save visualisations
# ---------------------------------------------------------------------------

def save_predictions(
    results: List[Dict],
    output_dir: Path,
) -> None:
    """Save predicted focus maps as PNG sequences."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available; skipping visualisation.")
        return

    for r in results:
        scene_dir = output_dir / r["scene"]
        scene_dir.mkdir(parents=True, exist_ok=True)

        pred = r["pred_focus_map"]  # (T, H, W) in [0, 1]
        start = r["start_frame"]

        for t in range(pred.shape[0]):
            frame_idx = start + t
            # Convert to grayscale uint8
            img = (pred[t] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(
                scene_dir / f"pred_focus_{frame_idx:04d}.png"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot wild evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--wild_data", type=str, required=True, help="Wild dataset root")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default="./wild_results")
    parser.add_argument("--window_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_preds", action="store_true", help="Save focus map PNGs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("precision", "bf16") == "bf16" and device.type == "cuda"

    wl = args.window_length or cfg.get("window_length", 8)
    resolution = tuple(cfg.get("resolution", [256, 256]))

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = cfg.get("model_type", "mamba")
    print(f"Model: {model_type}  |  Parameters: {n_params:,}")

    # Build wild dataloader
    print(f"Loading wild data from: {args.wild_data}")
    loader = build_wild_dataloader(
        data_root=args.wild_data,
        batch_size=args.batch_size,
        window_length=wl,
        stride=wl,  # non-overlapping for evaluation
        resolution=resolution,
    )
    print(f"  Clips: {len(loader.dataset)}")

    if len(loader.dataset) == 0:
        print("No valid wild scenes found. Check your data directory.")
        sys.exit(1)

    # Run inference
    results = run_inference(model, loader, device, use_amp)
    print(f"  Processed {len(results)} clips across {len(set(r['scene'] for r in results))} scenes")

    # Proxy metrics
    metrics = compute_proxy_metrics(results)
    if metrics:
        print("\nProxy Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("\nNo focus-distance labels found → qualitative evaluation only.")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_preds:
        print(f"Saving predictions to {output_dir}...")
        save_predictions(results, output_dir)
        print("Done saving predictions.")

    # Save metrics to JSON
    if metrics:
        import json
        metrics_path = output_dir / "wild_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    print("\nWild evaluation complete.")


if __name__ == "__main__":
    main()
