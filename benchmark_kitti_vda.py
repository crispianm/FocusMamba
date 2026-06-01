"""Evaluate VDA-S (metric and non-metric) on KITTI depth_selection val set.

Uses the val_selection_cropped split (1000 frames, 26 sequences) with
accumulated LiDAR ground truth.

Evaluation protocol follows the VDA paper:
  - Resize shorter edge to 518, both dims divisible by DINOv2 patch size (14)
  - Per-sequence scale-shift alignment in disparity space
    (fit s·(1/pred) + t ≈ 1/GT over all valid pixels in the sequence)
  - Depth cap 80m; min depth 1e-3m
  - Metrics: AbsRel, SqRel, RMSE, SILog, δ1, δ2, δ3

Paper reference (VDA-L, 500-frame eval):
  KITTI: AbsRel=0.083, δ1=0.944

Usage:
    cd /mnt/DATA/DegradedDepth/FocusMamba
    UV_PROJECT_ENVIRONMENT=/home/wg19671/.venvs/focusmamba \\
        uv run --no-sync python benchmark_kitti_vda.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.video_depth_anything_model import VideoDepthAnythingModel

KITTI_PATCH = 14          # DINOv2 patch size
KITTI_SHORT_EDGE = 518    # VDA paper inference resolution
MAX_DEPTH = 80.0
MIN_DEPTH = 1e-3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray:
    """Load RGB image → float32 HxWx3 in [0, 1]."""
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_gt_depth(path: Path) -> np.ndarray:
    """Load KITTI 16-bit GT depth PNG → float32 HxW in metres."""
    from PIL import Image
    gt = np.array(Image.open(path), dtype=np.uint16).astype(np.float32) / 256.0
    return gt  # 0 = invalid


def make_inference_size(h: int, w: int, short_edge: int, patch: int) -> tuple[int, int]:
    """Return (H, W) with shorter edge = short_edge and both divisible by patch."""
    if h <= w:
        new_h = short_edge
        new_w = int(round(w * short_edge / h))
    else:
        new_w = short_edge
        new_h = int(round(h * short_edge / w))
    # Round to nearest multiple of patch
    new_h = max(patch, round(new_h / patch) * patch)
    new_w = max(patch, round(new_w / patch) * patch)
    return new_h, new_w


def preprocess_frames(
    rgbs: list[np.ndarray],
    inf_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """HxWx3 float32 [0,1] list → (1, 3, T, H, W) raw tensor.

    The model applies ImageNet normalization internally, so we just resize.
    """
    frames = []
    for rgb in rgbs:
        t = torch.from_numpy(rgb).permute(2, 0, 1).to(device)          # (3,H,W)
        t = F.interpolate(t.unsqueeze(0), size=inf_hw, mode="bilinear",
                          align_corners=False).squeeze(0)               # (3,Hinf,Winf)
        frames.append(t)
    # Stack to (T, 3, Hinf, Winf) → permute → (3, T, Hinf, Winf) → add batch
    stacked = torch.stack(frames, dim=0)                                # (T, 3, H, W)
    return stacked.permute(1, 0, 2, 3).unsqueeze(0)                    # (1, 3, T, H, W)


# ---------------------------------------------------------------------------
# Alignment and metrics (NumPy, per-sequence global scale-shift)
# ---------------------------------------------------------------------------

def align_seq_vda(
    preds: list[np.ndarray],
    gts:   list[np.ndarray],
    valid: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Per-sequence scale-shift alignment for affine-invariant (non-metric) VDA.

    The model output is treated as an arbitrary-scale "depth-like" map.
    Solves s·pred + t ≈ 1/GT over ALL valid pixels in the sequence
    (one shared s,t for the whole sequence), then converts: depth = 1/(s·pred+t).

    This is the VDA paper protocol: "align video inverse depth with GT inverse depth"
    where the model output is the "video inverse depth".
    """
    pred_all, disp_g_all = [], []
    for pred, gt, msk in zip(preds, gts, valid):
        pred_all.append(pred[msk].clip(0.0))
        disp_g_all.append(1.0 / gt[msk].clip(MIN_DEPTH))

    dp = np.concatenate(pred_all)
    dg = np.concatenate(disp_g_all)

    A = np.stack([dp, np.ones_like(dp)], axis=1)
    result, _, _, _ = np.linalg.lstsq(A, dg, rcond=None)
    s, t = float(result[0]), float(result[1])

    aligned = []
    for pred in preds:
        aligned_disp = s * pred + t
        aligned.append(1.0 / np.maximum(aligned_disp, MIN_DEPTH))
    return aligned


def align_seq_depth(
    preds: list[np.ndarray],
    gts:   list[np.ndarray],
    valid: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Per-sequence depth-space alignment for metric models.
    Solves s·pred + t ≈ GT over all valid pixels in the sequence.
    """
    pred_all, gt_all = [], []
    for pred, gt, msk in zip(preds, gts, valid):
        pred_all.append(pred[msk])
        gt_all.append(gt[msk])

    dp = np.concatenate(pred_all)
    dg = np.concatenate(gt_all)

    A = np.stack([dp, np.ones_like(dp)], axis=1)
    result, _, _, _ = np.linalg.lstsq(A, dg, rcond=None)
    s, t = float(result[0]), float(result[1])

    aligned = []
    for pred in preds:
        aligned.append(np.maximum(s * pred + t, MIN_DEPTH))
    return aligned


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    max_depth: float = MAX_DEPTH,
) -> dict[str, float]:
    """AbsRel, SqRel, RMSE, SILog, δ1/2/3 on valid GT pixels ≤ max_depth."""
    msk = valid & (gt > MIN_DEPTH) & (gt <= max_depth)
    if msk.sum() == 0:
        return {}
    p = pred[msk].clip(MIN_DEPTH, max_depth)
    g = gt[msk].clip(MIN_DEPTH, max_depth)

    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel  = float(np.mean(((p - g) ** 2) / g))
    rmse    = float(np.sqrt(np.mean((p - g) ** 2)))

    d = np.log(p) - np.log(g)
    si_log = float(np.sqrt(np.mean(d ** 2) - np.mean(d) ** 2) * 100.0)

    ratio = np.maximum(p / g, g / p)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))

    return dict(abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse,
                si_log=si_log, delta1=d1, delta2=d2, delta3=d3)


# ---------------------------------------------------------------------------
# Sequence parsing
# ---------------------------------------------------------------------------

def parse_val_set(
    image_dir: Path, gt_dir: Path
) -> dict[str, list[tuple[Path, Path, int]]]:
    """Return {seq_id: [(image_path, gt_path, frame_idx), ...]} sorted by frame."""
    pattern = re.compile(
        r"(\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(\d+)_(image_\d+)\.png"
    )
    seqs: dict[str, list] = defaultdict(list)
    for img_path in sorted(image_dir.glob("*.png")):
        m = pattern.match(img_path.name)
        if not m:
            continue
        drive, frame_str, cam = m.group(1), m.group(2), m.group(3)
        gt_name = f"{drive}_groundtruth_depth_{frame_str}_{cam}.png"
        gt_path = gt_dir / gt_name
        if not gt_path.exists():
            continue
        seq_id = f"{drive}_{cam}"
        seqs[seq_id].append((img_path, gt_path, int(frame_str)))

    for seq in seqs.values():
        seq.sort(key=lambda x: x[2])
    return dict(seqs)


# ---------------------------------------------------------------------------
# Inference (chunked for long sequences)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_sequence(
    model: VideoDepthAnythingModel,
    rgbs: list[np.ndarray],
    inf_hw: tuple[int, int],
    orig_hw: tuple[int, int],
    device: torch.device,
    chunk_size: int = 4,   # small chunks: attention is O(T * N_patches^2)
) -> list[np.ndarray]:
    """Run model on a sequence in chunks; return list of HxW depth arrays."""
    preds = []
    for start in range(0, len(rgbs), chunk_size):
        chunk = rgbs[start : start + chunk_size]
        inp = preprocess_frames(chunk, inf_hw, device)   # (1, 3, T, H_inf, W_inf)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            out = model(inp)["depth"]  # (B, 1, T, H_inf, W_inf) or (B, T, H_inf, W_inf)
        # Normalise to (T, H_inf, W_inf) regardless of exact output shape
        out_t = out.squeeze(0)         # remove batch dim → (..., H, W)
        while out_t.ndim > 3:
            out_t = out_t.squeeze(0)   # strip leading size-1 dims until (T, H, W)
        T = out_t.shape[0]
        out_4d = out_t.unsqueeze(1).float()              # (T, 1, H_inf, W_inf)
        out_rs = F.interpolate(out_4d, size=orig_hw, mode="bilinear",
                               align_corners=False).squeeze(1)  # (T, H, W)
        for t in range(T):
            preds.append(out_rs[t].cpu().numpy())
    return preds


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    *,
    model: VideoDepthAnythingModel,
    model_name: str,
    seqs: dict[str, list],
    device: torch.device,
    inf_hw: tuple[int, int],
    orig_hw: tuple[int, int],
    per_frame_results: list[dict],
) -> dict[str, float]:
    """Evaluate model on all sequences. Returns aggregated metrics."""
    all_metrics: dict[str, list[float]] = defaultdict(list)

    for seq_idx, (seq_id, frames) in enumerate(sorted(seqs.items())):
        rgbs, gt_arrs, valids = [], [], []
        for img_path, gt_path, _ in frames:
            rgb = load_rgb(img_path)
            gt  = load_gt_depth(gt_path)
            valid = (gt > MIN_DEPTH) & (gt <= MAX_DEPTH)
            rgbs.append(rgb)
            gt_arrs.append(gt)
            valids.append(valid)

        if not any(v.any() for v in valids):
            continue

        # Inference
        preds_raw = infer_sequence(model, rgbs, inf_hw, orig_hw, device)

        # Per-sequence global alignment
        is_metric = getattr(model, "is_metric", True)
        if is_metric:
            preds_aligned = align_seq_depth(preds_raw, gt_arrs, valids)
        else:
            preds_aligned = align_seq_vda(preds_raw, gt_arrs, valids)

        # Per-frame metrics
        for i, (pred_a, gt_a, valid_a, (img_path, _, frame_idx)) in enumerate(
            zip(preds_aligned, gt_arrs, valids, frames)
        ):
            m = compute_metrics(pred_a, gt_a, valid_a)
            if not m:
                continue
            for k, v in m.items():
                all_metrics[k].append(v)
            per_frame_results.append({
                "model": model_name,
                "seq": seq_id,
                "frame": frame_idx,
                **{k: f"{v:.6f}" for k, v in m.items()},
            })

        if (seq_idx + 1) % 5 == 0 or (seq_idx + 1) == len(seqs):
            current = {k: np.mean(v) for k, v in all_metrics.items()}
            print(
                f"  [{seq_idx+1:2d}/{len(seqs)}] abs_rel={current.get('abs_rel',0):.4f} "
                f"delta1={current.get('delta1',0):.4f} si_log={current.get('si_log',0):.2f}"
            )

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kitti-dir", type=Path,
                   default=REPO_ROOT.parent / "data/kitti/depth_selection/val_selection_cropped")
    p.add_argument("--output-dir", type=Path, default=Path("runs/kitti_eval"))
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="List of checkpoint paths. Defaults to both stock checkpoints.")
    p.add_argument("--is-metric", choices=["auto", "true", "false"], default="auto",
                   help="Override is_metric for all --checkpoints. 'auto' (default) infers "
                        "from the 'metric_' filename prefix. Use 'true' for a fine-tuned "
                        "metric checkpoint (best.pt) whose name lacks the prefix.")
    p.add_argument("--device", default="auto")
    p.add_argument("--short-edge", type=int, default=KITTI_SHORT_EDGE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    image_dir = args.kitti_dir / "image"
    gt_dir    = args.kitti_dir / "groundtruth_depth"

    print("Parsing val_selection_cropped …")
    seqs = parse_val_set(image_dir, gt_dir)
    total_frames = sum(len(v) for v in seqs.values())
    print(f"  {len(seqs)} sequences, {total_frames} frames total")

    # Sample one image to get native resolution
    sample_img = next(iter(seqs.values()))[0][0]
    sample_rgb = load_rgb(sample_img)
    orig_h, orig_w = sample_rgb.shape[:2]
    inf_h, inf_w = make_inference_size(orig_h, orig_w, args.short_edge, KITTI_PATCH)
    print(f"  Native: {orig_h}×{orig_w}  →  Inference: {inf_h}×{inf_w}")

    checkpoints = args.checkpoints or [
        "checkpoints/video_depth_anything_vits.pth",      # non-metric
        "checkpoints/metric_video_depth_anything_vits.pth",  # metric
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_per_frame = []

    for ckpt_path_str in checkpoints:
        ckpt_path = Path(ckpt_path_str)
        if args.is_metric == "auto":
            is_metric = ckpt_path.name.startswith("metric_")
        else:
            is_metric = args.is_metric == "true"
        model_name = ckpt_path.stem
        label = "metric" if is_metric else "non-metric (VDA paper protocol)"
        print(f"\n=== {model_name} [{label}] ===")

        model = VideoDepthAnythingModel(
            variant="small",
            num_frames=32,
            checkpoint_path=str(ckpt_path),
            strict_checkpoint=False,
            mode="offline",
        ).to(device).eval()
        model.is_metric = is_metric

        t0 = time.time()
        agg = evaluate(
            model=model,
            model_name=model_name,
            seqs=seqs,
            device=device,
            inf_hw=(inf_h, inf_w),
            orig_hw=(orig_h, orig_w),
            per_frame_results=all_per_frame,
        )
        elapsed = time.time() - t0

        print(f"\n  ── {model_name} ──")
        print(f"  AbsRel:  {agg['abs_rel']:.4f}   (VDA-L paper: 0.083)")
        print(f"  δ1:      {agg['delta1']:.4f}   (VDA-L paper: 0.944)")
        print(f"  SILog:   {agg['si_log']:.3f}")
        print(f"  RMSE:    {agg['rmse']:.3f}")
        print(f"  Time:    {elapsed:.1f}s  ({total_frames / elapsed:.1f} fps)")

        summary_rows.append({
            "model": model_name,
            "n_frames": total_frames,
            "inf_res": f"{inf_h}x{inf_w}",
            "elapsed_s": f"{elapsed:.1f}",
            **{k: f"{v:.6f}" for k, v in agg.items()},
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save outputs
    summary_path = args.output_dir / "kitti_vda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    tsv_path = args.output_dir / "kitti_vda_per_frame.tsv"
    if all_per_frame:
        with open(tsv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_per_frame[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(all_per_frame)

    print(f"\nResults saved to {args.output_dir}/")

    # Final comparison table
    print("\n" + "=" * 60)
    print(f"{'Model':<45} {'AbsRel':>8} {'δ1':>8} {'SILog':>8}")
    print("-" * 60)
    for row in summary_rows:
        print(f"{row['model']:<45} {float(row['abs_rel']):>8.4f} "
              f"{float(row['delta1']):>8.4f} {float(row['si_log']):>8.3f}")
    print(f"{'VDA-L (paper, 500-frame sequences)':<45} {'0.083':>8} {'0.944':>8} {'-':>8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
