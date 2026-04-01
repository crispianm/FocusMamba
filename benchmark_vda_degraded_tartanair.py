"""Benchmark ViT-S and ViT-L metric VDA checkpoints on degraded TartanAirV2.

This is the script version of the benchmarking section added to `zz_dataset.ipynb`.
It evaluates the requested checkpoints on degraded TartanAirV2 validation clips,
reports pixel-weighted `abs_rel` and `delta1`, and saves outputs in the repo root.

Typical HPC usage:
    srun --gpus=1 --time=02:00:00 --pty /bin/bash --login
    cd /projects/b5dh/FocusMamba
    source .venv/bin/activate
    python benchmark_vda_degraded_tartanair.py --device cuda --num-eval-clips 32
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path("/projects/b5dh/FocusMamba")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.degradation import build_degradation
from dataloader.tartanair_v2 import TartanAirV2Dataset
from models.video_depth_anything_model import VideoDepthAnythingModel


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/experiments/generated/focused_redirection_20260310/05_vdal_degraded_stream.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT
DEFAULT_PREFIX = "vda_degraded_tartanair"
DEFAULT_CHECKPOINTS = {
    "vits": REPO_ROOT / "checkpoints/metric_video_depth_anything_vits.pth",
    "vitl": REPO_ROOT / "checkpoints/metric_video_depth_anything_vitl.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--models", default="vits,vitl", help="Comma-separated subset of: vits,vitl"
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument(
        "--mode", choices=("offline", "streaming_emulated"), default="offline"
    )
    parser.add_argument("--num-eval-clips", type=int, default=8)
    parser.add_argument("--num-display-clips", type=int, default=2)
    parser.add_argument(
        "--image-size",
        default="518,518",
        help="Dataset resize H,W. Defaults to 518,518 to match VDA training resolution.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional fine-tuned checkpoint to benchmark instead of the stock VDA weights.",
    )
    parser.add_argument(
        "--checkpoint-label",
        default=None,
        help="Label for --checkpoint-path in summaries. Defaults to the checkpoint stem.",
    )
    parser.add_argument(
        "--checkpoint-variant",
        default=None,
        help="Override model variant for --checkpoint-path. Defaults to config.model.variant.",
    )
    parser.add_argument(
        "--inference-size",
        default=None,
        help="Optional H,W model inference size. Example: 392,392",
    )
    return parser.parse_args()


def parse_models(raw_models: str) -> list[str]:
    names = [part.strip().lower() for part in raw_models.split(",") if part.strip()]
    if not names:
        raise ValueError("No models requested.")
    unknown = [name for name in names if name not in DEFAULT_CHECKPOINTS]
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(sorted(set(unknown)))}")
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def resolve_model_specs(
    args: argparse.Namespace, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    if args.checkpoint_path is not None:
        variant = str(
            args.checkpoint_variant or cfg.get("model", {}).get("variant", "small")
        ).strip()
        label = str(args.checkpoint_label or args.checkpoint_path.stem).strip()
        if not label:
            label = args.checkpoint_path.stem
        return [
            {
                "name": label,
                "variant": variant,
                "checkpoint_path": args.checkpoint_path,
            }
        ]

    return [
        {
            "name": model_name,
            "variant": model_name,
            "checkpoint_path": DEFAULT_CHECKPOINTS[model_name],
        }
        for model_name in parse_models(args.models)
    ]


def parse_hw(
    raw_hw: str | None, *, arg_name: str = "--inference-size"
) -> tuple[int, int] | None:
    if raw_hw is None:
        return None
    normalized = raw_hw.lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) != 2:
        raise ValueError(
            f"{arg_name} must look like H,W, HxW, or a single square size. Got {raw_hw!r}"
        )
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{arg_name} must use positive integers. Got {raw_hw!r}")
    return (height, width)


def resolve_device(raw_device: str) -> str:
    if raw_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw_device


def resize_video_tensor(
    tensor: torch.Tensor,
    *,
    size: tuple[int, int],
    mode: str,
    align_corners: bool | None,
) -> torch.Tensor:
    batch_size, channels, num_frames, height, width = tensor.shape
    flat = tensor.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, channels, height, width
    )
    kwargs: dict[str, Any] = {"size": size, "mode": mode}
    if align_corners is not None:
        kwargs["align_corners"] = align_corners
    resized = F.interpolate(flat, **kwargs)
    return (
        resized.reshape(batch_size, num_frames, channels, size[0], size[1])
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def init_metric_state() -> dict[str, float | int]:
    return {"valid_pixels": 0, "sum_abs_rel": 0.0, "delta1_hits": 0}


def update_metric_state(
    state: dict[str, float | int],
    *,
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float,
) -> None:
    valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(gt)
    valid = valid & (pred > min_depth) & (gt > min_depth) & (gt < max_depth)
    if not valid.any():
        return

    pred_valid = pred[valid].float()
    gt_valid = gt[valid].float()
    ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)

    state["valid_pixels"] = int(state["valid_pixels"]) + int(pred_valid.numel())
    state["sum_abs_rel"] = float(state["sum_abs_rel"]) + float(
        ((pred_valid - gt_valid).abs() / gt_valid).sum().item()
    )
    state["delta1_hits"] = int(state["delta1_hits"]) + int((ratio < 1.25).sum().item())


def finalize_metric_state(state: dict[str, float | int]) -> dict[str, float | int]:
    if int(state["valid_pixels"]) == 0:
        return {"valid_pixels": 0, "abs_rel": float("nan"), "delta1": float("nan")}
    denom = float(state["valid_pixels"])
    return {
        "valid_pixels": int(state["valid_pixels"]),
        "abs_rel": float(state["sum_abs_rel"]) / denom,
        "delta1": int(state["delta1_hits"]) / denom,
    }


def compute_clip_metrics(
    *,
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    max_depth: float,
) -> dict[str, float | int]:
    state = init_metric_state()
    update_metric_state(state, pred=pred, gt=gt, mask=mask, max_depth=max_depth)
    return finalize_metric_state(state)


def load_dataset(
    config_path: Path,
    *,
    image_hw: tuple[int, int] | None,
) -> tuple[dict[str, Any], TartanAirV2Dataset]:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    data_cfg = cfg["data"]
    degradation = build_degradation(cfg)
    if degradation is None:
        raise RuntimeError(f"{config_path} does not enable degradation.")

    dataset_hw = image_hw
    if dataset_hw is None:
        dataset_hw = tuple(int(v) for v in data_cfg.get("image_size", (256, 256)))

    dataset = TartanAirV2Dataset(
        root=str(data_cfg["root"]),
        num_frames=int(data_cfg.get("val_num_frames", data_cfg.get("num_frames", 8))),
        image_size=dataset_hw,
        max_trajectories=data_cfg.get(
            "max_val_trajectories", data_cfg.get("max_trajectories")
        ),
        clip_stride=int(data_cfg.get("clip_stride", 8)),
        frame_stride=int(data_cfg.get("frame_stride", 1)),
        split="val",
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=int(data_cfg.get("seed", 42)),
        difficulty=str(data_cfg.get("difficulty", "Data_easy")),
        camera=str(data_cfg.get("camera", "lcam_front")),
        max_depth=float(data_cfg.get("max_depth", 80.0)),
        envs=data_cfg.get("envs"),
        teacher_cache_dir=None,
        degradation=degradation,
        return_clean_and_degraded=True,
        return_clean_reference=False,
    )
    return cfg, dataset


def load_model(
    *,
    variant: str,
    checkpoint_path: Path,
    num_frames: int,
    mode: str,
    device: str,
) -> VideoDepthAnythingModel:
    model = VideoDepthAnythingModel(
        variant=variant,
        num_frames=max(32, int(num_frames)),
        checkpoint_path=str(checkpoint_path),
        strict_checkpoint=False,
        mode=mode,
    )
    return model.to(device).eval()


@torch.inference_mode()
def predict_depth(
    model: VideoDepthAnythingModel,
    *,
    frames: torch.Tensor,
    device: str,
    mode: str,
    inference_hw: tuple[int, int] | None,
) -> torch.Tensor:
    original_hw = tuple(int(v) for v in frames.shape[-2:])
    model_frames = frames
    if inference_hw is not None and tuple(original_hw) != tuple(inference_hw):
        model_frames = resize_video_tensor(
            model_frames,
            size=inference_hw,
            mode="bilinear",
            align_corners=False,
        )

    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(model_frames, mode=mode)["depth"]
    else:
        pred = model(model_frames, mode=mode)["depth"]

    if (
        inference_hw is not None
        and tuple(int(v) for v in pred.shape[-2:]) != original_hw
    ):
        pred = resize_video_tensor(
            pred,
            size=original_hw,
            mode="bilinear",
            align_corners=False,
        )
    return pred.float()


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_clip_examples(eval_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clip_examples: list[dict[str, Any]] = []
    for item in eval_items:
        mid_idx = int(item["depth"].shape[1] // 2)
        clip_examples.append(
            {
                "video_id": item["video_id"],
                "start_frame": int(item["start_frame"]),
                "degraded_rgb": item["degraded_frames"][:, mid_idx]
                .permute(1, 2, 0)
                .cpu()
                .numpy(),
                "gt_depth": item["depth"][0, mid_idx].cpu().numpy(),
                "gt_mask": item["mask"][0, mid_idx].cpu().numpy().astype(bool),
                "predictions": {},
                "metrics": {},
            }
        )
    return clip_examples


def compute_abs_rel_heatmap(
    *,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_mask: np.ndarray,
    max_depth: float,
    min_depth: float = 1e-3,
) -> np.ndarray:
    valid = (
        gt_mask.astype(bool)
        & np.isfinite(pred_depth)
        & np.isfinite(gt_depth)
        & (pred_depth > min_depth)
        & (gt_depth > min_depth)
        & (gt_depth < max_depth)
    )
    heatmap = np.full(gt_depth.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        heatmap[valid] = np.abs(pred_depth[valid] - gt_depth[valid]) / np.maximum(
            gt_depth[valid], min_depth
        )
    return heatmap


def save_figure(
    *,
    output_path: Path,
    display_examples: list[dict[str, Any]],
    model_names: list[str],
    max_depth: float,
    total_clips: int,
    device: str,
) -> None:
    if not display_examples:
        return

    fig, axes = plt.subplots(
        len(display_examples),
        2 + len(model_names),
        figsize=(4.4 * (2 + len(model_names)), 3.6 * len(display_examples)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, example in enumerate(display_examples):
        rgb = np.clip(example["degraded_rgb"], 0.0, 1.0)
        gt_depth = np.clip(example["gt_depth"], 0.0, max_depth)
        valid_gt = gt_depth[gt_depth > 0]
        vmax = float(np.percentile(valid_gt, 95)) if valid_gt.size else max_depth

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title("Degraded RGB")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_ylabel(
            f"{example['video_id']}\nstart={example['start_frame']}",
            rotation=0,
            labelpad=60,
            va="center",
        )

        gt_im = axes[row_idx, 1].imshow(gt_depth, cmap="magma_r", vmin=0, vmax=vmax)
        axes[row_idx, 1].set_title("GT depth")
        axes[row_idx, 1].axis("off")
        fig.colorbar(gt_im, ax=axes[row_idx, 1], fraction=0.046, pad=0.02)

        for col_idx, model_name in enumerate(model_names, start=2):
            pred_depth = np.clip(example["predictions"][model_name], 0.0, max_depth)
            pred_im = axes[row_idx, col_idx].imshow(
                pred_depth, cmap="magma_r", vmin=0, vmax=vmax
            )
            metrics = example["metrics"][model_name]
            axes[row_idx, col_idx].set_title(
                f"{model_name}\nabs_rel={metrics['abs_rel']:.4f}  delta1={metrics['delta1']:.4f}"
            )
            axes[row_idx, col_idx].axis("off")
            fig.colorbar(pred_im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.02)

    fig.suptitle(
        f"Degraded TartanAirV2 validation samples ({len(display_examples)} of {total_clips} clips, device={device})",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_absrel_gap_rows(clip_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in clip_examples:
        if "vits" not in example["metrics"] or "vitl" not in example["metrics"]:
            continue
        vits_metrics = example["metrics"]["vits"]
        vitl_metrics = example["metrics"]["vitl"]
        rows.append(
            {
                "video_id": example["video_id"],
                "start_frame": example["start_frame"],
                "vits_abs_rel": float(vits_metrics["abs_rel"]),
                "vitl_abs_rel": float(vitl_metrics["abs_rel"]),
                "vits_delta1": float(vits_metrics["delta1"]),
                "vitl_delta1": float(vitl_metrics["delta1"]),
                "abs_rel_gap_vitl_minus_vits": float(vitl_metrics["abs_rel"])
                - float(vits_metrics["abs_rel"]),
                "delta1_gap_vitl_minus_vits": float(vitl_metrics["delta1"])
                - float(vits_metrics["delta1"]),
            }
        )
    rows.sort(key=lambda row: row["abs_rel_gap_vitl_minus_vits"])
    return rows


def save_absrel_gap_figure(
    *,
    output_path: Path,
    selected_examples: list[dict[str, Any]],
    max_depth: float,
) -> None:
    if not selected_examples:
        return

    error_values: list[np.ndarray] = []
    for example in selected_examples:
        for model_name in ("vits", "vitl"):
            heatmap = compute_abs_rel_heatmap(
                pred_depth=np.asarray(
                    example["predictions"][model_name], dtype=np.float32
                ),
                gt_depth=np.asarray(example["gt_depth"], dtype=np.float32),
                gt_mask=np.asarray(example["gt_mask"], dtype=bool),
                max_depth=max_depth,
            )
            valid = np.isfinite(heatmap)
            if np.any(valid):
                error_values.append(heatmap[valid])

    if error_values:
        vmax_error = float(np.percentile(np.concatenate(error_values), 95))
        vmax_error = max(vmax_error, 0.25)
    else:
        vmax_error = 1.0

    fig, axes = plt.subplots(
        len(selected_examples),
        6,
        figsize=(25, 4.2 * len(selected_examples)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, example in enumerate(selected_examples):
        rgb = np.clip(example["degraded_rgb"], 0.0, 1.0)
        gt_depth = np.clip(example["gt_depth"], 0.0, max_depth)
        gt_mask = np.asarray(example["gt_mask"], dtype=bool)
        valid_gt = gt_depth[gt_mask & (gt_depth > 0)]
        vmax_depth = float(np.percentile(valid_gt, 95)) if valid_gt.size else max_depth
        gap = float(example["metrics"]["vitl"]["abs_rel"]) - float(
            example["metrics"]["vits"]["abs_rel"]
        )

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title("Degraded RGB")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_ylabel(
            f"{example['video_id']}\nstart={example['start_frame']}\ngap={gap:+.4f}",
            rotation=0,
            labelpad=70,
            va="center",
        )

        gt_im = axes[row_idx, 1].imshow(
            gt_depth, cmap="magma_r", vmin=0, vmax=vmax_depth
        )
        axes[row_idx, 1].set_title("GT depth")
        axes[row_idx, 1].axis("off")
        fig.colorbar(gt_im, ax=axes[row_idx, 1], fraction=0.046, pad=0.02)

        for base_col, model_name in ((2, "vits"), (4, "vitl")):
            pred_depth = np.clip(example["predictions"][model_name], 0.0, max_depth)
            pred_im = axes[row_idx, base_col].imshow(
                pred_depth, cmap="magma_r", vmin=0, vmax=vmax_depth
            )
            metrics = example["metrics"][model_name]
            axes[row_idx, base_col].set_title(
                f"{model_name}\nabs_rel={metrics['abs_rel']:.4f}  delta1={metrics['delta1']:.4f}"
            )
            axes[row_idx, base_col].axis("off")
            fig.colorbar(pred_im, ax=axes[row_idx, base_col], fraction=0.046, pad=0.02)

            heatmap = compute_abs_rel_heatmap(
                pred_depth=np.asarray(
                    example["predictions"][model_name], dtype=np.float32
                ),
                gt_depth=np.asarray(example["gt_depth"], dtype=np.float32),
                gt_mask=gt_mask,
                max_depth=max_depth,
            )
            heatmap_im = axes[row_idx, base_col + 1].imshow(
                np.ma.masked_invalid(heatmap),
                cmap="viridis",
                vmin=0.0,
                vmax=vmax_error,
            )
            axes[row_idx, base_col + 1].set_title(f"{model_name} abs_rel heatmap")
            axes[row_idx, base_col + 1].axis("off")
            fig.colorbar(
                heatmap_im, ax=axes[row_idx, base_col + 1], fraction=0.046, pad=0.02
            )

    fig.suptitle("Best and worst clips by abs_rel gap (vitl - vits)", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_scatter_plot(
    *,
    output_path: Path,
    clip_examples: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = {"vits": "#1f77b4", "vitl": "#d62728"}
    labels_done: set[str] = set()

    for example in clip_examples:
        for model_name in ("vits", "vitl"):
            if model_name not in example["metrics"]:
                continue
            metrics = example["metrics"][model_name]
            label = model_name if model_name not in labels_done else None
            ax.scatter(
                float(metrics["abs_rel"]),
                float(metrics["delta1"]),
                s=28,
                alpha=0.75,
                color=colors[model_name],
                label=label,
            )
            labels_done.add(model_name)

    for row in gap_rows[:4] + gap_rows[-4:]:
        clip_label = f"{row['start_frame']}"
        ax.annotate(
            clip_label,
            (row["vitl_abs_rel"], row["vitl_delta1"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color=colors["vitl"],
        )
        ax.annotate(
            clip_label,
            (row["vits_abs_rel"], row["vits_delta1"]),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=8,
            color=colors["vits"],
        )

    ax.set_title("Per-clip abs_rel vs delta1")
    ax.set_xlabel("abs_rel (lower is better)")
    ax.set_ylabel("delta1 (higher is better)")
    ax.grid(alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("")
    print("Summary")
    print("model\tclips\tabs_rel\tdelta1\tseconds\tdevice\tmode")
    for row in rows:
        print(
            f"{row['model']}\t{row['clips']}\t{row['abs_rel']:.4f}\t"
            f"{row['delta1']:.4f}\t{row['seconds']:.1f}\t{row['device']}\t{row['mode']}"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    image_hw = parse_hw(args.image_size, arg_name="--image-size")
    inference_hw = parse_hw(args.inference_size, arg_name="--inference-size")

    cfg, dataset = load_dataset(args.config, image_hw=image_hw)
    model_specs = resolve_model_specs(args, cfg)
    model_names = [str(spec["name"]) for spec in model_specs]
    max_depth = float(cfg["data"].get("max_depth", 80.0))
    num_eval_clips = min(max(int(args.num_eval_clips), 0), len(dataset))
    if num_eval_clips <= 0:
        raise ValueError("No evaluation clips selected.")
    num_display_clips = min(max(int(args.num_display_clips), 0), num_eval_clips)

    eval_items = [dataset[idx] for idx in range(num_eval_clips)]
    clip_examples = build_clip_examples(eval_items)
    display_examples = clip_examples[:num_display_clips]

    summary_rows: list[dict[str, Any]] = []
    per_clip_rows: list[dict[str, Any]] = []

    print(f"repo_root={REPO_ROOT}")
    print(f"config={args.config}")
    print(f"device={device}")
    print(f"mode={args.mode}")
    print(f"clips={num_eval_clips}/{len(dataset)}")
    print(f"image_size={dataset.image_size}")
    if inference_hw is not None:
        print(f"inference_hw={inference_hw}")
    for spec in model_specs:
        print(f"{spec['name']}_checkpoint={spec['checkpoint_path']}")

    wall_start = time.perf_counter()
    for spec in model_specs:
        model_name = str(spec["name"])
        checkpoint_path = Path(spec["checkpoint_path"])
        model = load_model(
            variant=str(spec["variant"]),
            checkpoint_path=checkpoint_path,
            num_frames=dataset.num_frames,
            mode=args.mode,
            device=device,
        )
        metric_state = init_metric_state()
        model_start = time.perf_counter()

        for clip_idx, item in enumerate(eval_items):
            frames = item["degraded_frames"].unsqueeze(0).to(device, non_blocking=True)
            gt = item["depth"].unsqueeze(0).to(device, non_blocking=True)
            mask = item["mask"].unsqueeze(0).to(device, non_blocking=True)

            pred = predict_depth(
                model,
                frames=frames,
                device=device,
                mode=args.mode,
                inference_hw=inference_hw,
            )
            update_metric_state(
                metric_state, pred=pred, gt=gt, mask=mask, max_depth=max_depth
            )
            clip_metrics = compute_clip_metrics(
                pred=pred, gt=gt, mask=mask, max_depth=max_depth
            )
            per_clip_rows.append(
                {
                    "model": model_name,
                    "video_id": item["video_id"],
                    "start_frame": int(item["start_frame"]),
                    "abs_rel": f"{clip_metrics['abs_rel']:.6f}",
                    "delta1": f"{clip_metrics['delta1']:.6f}",
                    "valid_pixels": int(clip_metrics["valid_pixels"]),
                }
            )

            clip_examples[clip_idx]["predictions"][model_name] = (
                pred[0, 0, pred.shape[2] // 2].detach().cpu().numpy()
            )
            clip_examples[clip_idx]["metrics"][model_name] = clip_metrics

        elapsed = time.perf_counter() - model_start
        metrics = finalize_metric_state(metric_state)
        summary_rows.append(
            {
                "model": model_name,
                "checkpoint": checkpoint_path.name,
                "clips": num_eval_clips,
                "abs_rel": float(metrics["abs_rel"]),
                "delta1": float(metrics["delta1"]),
                "valid_pixels": int(metrics["valid_pixels"]),
                "seconds": elapsed,
                "device": device,
                "mode": args.mode,
            }
        )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    summary_rows.sort(key=lambda row: row["abs_rel"])
    per_clip_rows.sort(
        key=lambda row: (row["model"], row["video_id"], row["start_frame"])
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / args.output_prefix
    summary_tsv = base.with_name(f"{base.name}_summary.tsv")
    per_clip_tsv = base.with_name(f"{base.name}_per_clip.tsv")
    summary_json = base.with_name(f"{base.name}_summary.json")
    figure_png = base.with_name(f"{base.name}_examples.png")
    gap_tsv = base.with_name(f"{base.name}_absrel_gap.tsv")
    gap_png = base.with_name(f"{base.name}_absrel_gap_diagnostics.png")
    scatter_png = base.with_name(f"{base.name}_absrel_vs_delta1_scatter.png")

    summary_write_rows = []
    for row in summary_rows:
        summary_write_rows.append(
            {
                "model": row["model"],
                "checkpoint": row["checkpoint"],
                "clips": row["clips"],
                "abs_rel": f"{row['abs_rel']:.6f}",
                "delta1": f"{row['delta1']:.6f}",
                "valid_pixels": row["valid_pixels"],
                "seconds": f"{row['seconds']:.3f}",
                "device": row["device"],
                "mode": row["mode"],
            }
        )

    write_tsv(
        summary_tsv,
        summary_write_rows,
        [
            "model",
            "checkpoint",
            "clips",
            "abs_rel",
            "delta1",
            "valid_pixels",
            "seconds",
            "device",
            "mode",
        ],
    )
    write_tsv(
        per_clip_tsv,
        per_clip_rows,
        ["model", "video_id", "start_frame", "abs_rel", "delta1", "valid_pixels"],
    )
    with summary_json.open("w") as handle:
        json.dump(
            {
                "config": str(args.config),
                "device": device,
                "mode": args.mode,
                "image_hw": image_hw,
                "num_eval_clips": num_eval_clips,
                "num_display_clips": num_display_clips,
                "inference_hw": inference_hw,
                "total_elapsed_seconds": round(time.perf_counter() - wall_start, 3),
                "summary": summary_write_rows,
            },
            handle,
            indent=2,
        )
    save_figure(
        output_path=figure_png,
        display_examples=display_examples,
        model_names=model_names,
        max_depth=max_depth,
        total_clips=num_eval_clips,
        device=device,
    )

    gap_rows: list[dict[str, Any]] = []
    if "vits" in model_names and "vitl" in model_names:
        gap_rows = build_absrel_gap_rows(clip_examples)
        gap_write_rows = []
        for row in gap_rows:
            gap_write_rows.append(
                {
                    "video_id": row["video_id"],
                    "start_frame": row["start_frame"],
                    "vits_abs_rel": f"{row['vits_abs_rel']:.6f}",
                    "vitl_abs_rel": f"{row['vitl_abs_rel']:.6f}",
                    "vits_delta1": f"{row['vits_delta1']:.6f}",
                    "vitl_delta1": f"{row['vitl_delta1']:.6f}",
                    "abs_rel_gap_vitl_minus_vits": f"{row['abs_rel_gap_vitl_minus_vits']:.6f}",
                    "delta1_gap_vitl_minus_vits": f"{row['delta1_gap_vitl_minus_vits']:.6f}",
                }
            )
        write_tsv(
            gap_tsv,
            gap_write_rows,
            [
                "video_id",
                "start_frame",
                "vits_abs_rel",
                "vitl_abs_rel",
                "vits_delta1",
                "vitl_delta1",
                "abs_rel_gap_vitl_minus_vits",
                "delta1_gap_vitl_minus_vits",
            ],
        )

        example_lookup = {
            (example["video_id"], int(example["start_frame"])): example
            for example in clip_examples
        }
        selected_examples = []
        for row in gap_rows[:4] + gap_rows[-4:]:
            key = (row["video_id"], int(row["start_frame"]))
            if key in example_lookup:
                selected_examples.append(example_lookup[key])

        save_absrel_gap_figure(
            output_path=gap_png,
            selected_examples=selected_examples,
            max_depth=max_depth,
        )
        save_scatter_plot(
            output_path=scatter_png,
            clip_examples=clip_examples,
            gap_rows=gap_rows,
        )

    print_summary(summary_rows)
    print("")
    print(f"total_elapsed={time.perf_counter() - wall_start:.1f}s")
    print(f"saved_summary_tsv={summary_tsv}")
    print(f"saved_per_clip_tsv={per_clip_tsv}")
    print(f"saved_summary_json={summary_json}")
    if num_display_clips > 0:
        print(f"saved_examples_png={figure_png}")
    if gap_rows:
        print(f"saved_absrel_gap_tsv={gap_tsv}")
        print(f"saved_absrel_gap_diagnostics_png={gap_png}")
        print(f"saved_absrel_vs_delta1_scatter_png={scatter_png}")


if __name__ == "__main__":
    main()
