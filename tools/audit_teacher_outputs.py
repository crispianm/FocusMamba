#!/usr/bin/env python
"""Audit heterogeneous teacher outputs on clean TartanAir validation clips."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader.tartanair_v2 import TartanAirV2Dataset
from evaluation.metrics.temporal_metrics import frame_depth_variation
from models.teachers import build_teacher
from training.trainer import compute_depth_metrics


DEFAULT_TEACHERS = [
    {
        "name": "video_depth_anything",
        "enabled": True,
        "checkpoint_path": "checkpoints/metric_video_depth_anything_vitl.pth",
        "device": "cuda",
        "temporal_window": 8,
        "weight": 1.0,
    },
    {
        "name": "depth_anything_v3",
        "enabled": True,
        "checkpoint_path": "checkpoints/da3_metric.safetensors",
        "device": "cuda",
        "weight": 1.0,
    },
    {
        "name": "depth_pro",
        "enabled": True,
        "checkpoint_path": "checkpoints/depth_pro.pt",
        "device": "cuda",
        "weight": 1.0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment YAML used to define the validation dataset.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to store audit summaries."
    )
    parser.add_argument(
        "--num-clips", type=int, default=64, help="Number of validation clips to audit."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for auditing."
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader workers."
    )
    parser.add_argument(
        "--device", default=None, help="Override device, e.g. cuda or cpu."
    )
    parser.add_argument(
        "--teachers",
        nargs="*",
        default=None,
        help="Optional teacher subset, e.g. video_depth_anything depth_anything_v3 depth_pro",
    )
    return parser.parse_args()


def _resolve_teacher_cfgs(cfg: dict, teacher_subset: set[str] | None) -> list[dict]:
    teacher_cfgs = list(cfg.get("teachers", [])) or list(DEFAULT_TEACHERS)
    resolved = []
    for teacher_cfg in teacher_cfgs:
        if not teacher_cfg.get("enabled", True):
            continue
        name = teacher_cfg["name"]
        if teacher_subset and name not in teacher_subset:
            continue
        resolved.append(dict(teacher_cfg))
    return resolved


def _build_val_dataset(cfg: dict) -> TartanAirV2Dataset:
    data_cfg = cfg.get("data", {})
    train_frames = int(data_cfg.get("train_num_frames", data_cfg.get("num_frames", 8)))
    val_frames = int(
        data_cfg.get("val_num_frames", data_cfg.get("num_frames", train_frames))
    )
    image_size = tuple(int(v) for v in data_cfg.get("image_size", (256, 256)))
    return TartanAirV2Dataset(
        root=str(data_cfg["root"]),
        num_frames=val_frames,
        image_size=image_size,
        max_trajectories=data_cfg.get("max_trajectories"),
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
        degradation=None,
        return_clean_and_degraded=False,
    )


def _teacher_valid_mask(
    name: str, depth: torch.Tensor, gt_mask: torch.Tensor, max_depth: float
) -> torch.Tensor:
    valid = torch.isfinite(depth) & (depth > 0.1) & (depth <= max_depth)
    if name == "depth_anything_v3":
        valid = valid & (depth > 0)
    return valid & gt_mask.bool()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_subset = set(args.teachers) if args.teachers else None
    teacher_cfgs = _resolve_teacher_cfgs(cfg, teacher_subset)
    if not teacher_cfgs:
        raise ValueError("No enabled teachers resolved from config/arguments.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _build_val_dataset(cfg)
    if args.num_clips > 0 and len(dataset) > args.num_clips:
        dataset = Subset(dataset, list(range(args.num_clips)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    teachers = {
        t_cfg["name"]: build_teacher(t_cfg["name"], t_cfg, device=device)
        for t_cfg in teacher_cfgs
    }
    max_depth = float(cfg.get("data", {}).get("max_depth", 80.0))

    summary = defaultdict(lambda: defaultdict(list))
    pairwise = defaultdict(list)

    for batch in tqdm(loader, desc="Teacher audit", unit="clip"):
        frames = batch["frames"].to(device)
        gt_depth = batch["depth"].to(device)
        gt_mask = batch["mask"].to(device).bool()

        teacher_outputs: Dict[str, torch.Tensor] = {}
        teacher_valids: Dict[str, torch.Tensor] = {}
        for name, teacher in teachers.items():
            with torch.no_grad():
                depth = teacher.predict(frames).float()
            teacher_outputs[name] = depth
            valid = _teacher_valid_mask(name, depth, gt_mask, max_depth)
            teacher_valids[name] = valid

            clipped = depth >= max_depth
            zeros = depth <= 0
            finite_valid = depth[valid]
            metrics = compute_depth_metrics(depth, gt_depth, mask=valid)
            summary[name]["abs_rel"].append(metrics["abs_rel"])
            summary[name]["rmse"].append(metrics["rmse"])
            summary[name]["si_log"].append(metrics["si_log"])
            summary[name]["delta1"].append(metrics["delta1"])
            summary[name]["fdv"].append(
                frame_depth_variation(depth, mask=valid.float())
            )
            summary[name]["invalid_fraction"].append(
                float((~valid).float().mean().item())
            )
            summary[name]["zero_fraction"].append(float(zeros.float().mean().item()))
            summary[name]["clipped_fraction"].append(
                float(clipped.float().mean().item())
            )
            if finite_valid.numel() > 0:
                summary[name]["min_depth"].append(float(finite_valid.min().item()))
                summary[name]["max_depth"].append(float(finite_valid.max().item()))
                summary[name]["median_depth"].append(
                    float(finite_valid.median().item())
                )

        names = list(teacher_outputs.keys())
        for idx, name_i in enumerate(names):
            for name_j in names[idx + 1 :]:
                common = teacher_valids[name_i] & teacher_valids[name_j]
                if not common.any():
                    continue
                log_i = torch.log(teacher_outputs[name_i].clamp(min=1e-6))
                log_j = torch.log(teacher_outputs[name_j].clamp(min=1e-6))
                disagreement = (log_i[common] - log_j[common]).abs().mean().item()
                pairwise[(name_i, name_j)].append(float(disagreement))

    teacher_rows = []
    absrel_for_priors = {}
    for name, values in summary.items():
        row = {
            "teacher": name,
            "clips": len(values["abs_rel"]),
            "abs_rel": _mean(values["abs_rel"]),
            "rmse": _mean(values["rmse"]),
            "si_log": _mean(values["si_log"]),
            "delta1": _mean(values["delta1"]),
            "fdv": _mean(values["fdv"]),
            "invalid_fraction": _mean(values["invalid_fraction"]),
            "zero_fraction": _mean(values["zero_fraction"]),
            "clipped_fraction": _mean(values["clipped_fraction"]),
            "min_depth": min(values["min_depth"])
            if values["min_depth"]
            else float("nan"),
            "max_depth": max(values["max_depth"])
            if values["max_depth"]
            else float("nan"),
            "median_depth": _mean(values["median_depth"]),
        }
        teacher_rows.append(row)
        absrel_for_priors[name] = row["abs_rel"]

        with (output_dir / f"{name}_summary.json").open("w") as f:
            json.dump(row, f, indent=2)

    denom = sum(1.0 / (absrel + 1e-3) for absrel in absrel_for_priors.values())
    teacher_priors = {
        name: float((1.0 / (absrel + 1e-3)) / denom)
        for name, absrel in absrel_for_priors.items()
    }

    pairwise_rows = [
        {
            "teacher_a": name_a,
            "teacher_b": name_b,
            "mean_abs_log_diff": _mean(values),
            "clips": len(values),
        }
        for (name_a, name_b), values in sorted(pairwise.items())
    ]

    with (output_dir / "teacher_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(teacher_rows[0].keys()))
        writer.writeheader()
        writer.writerows(teacher_rows)

    with (output_dir / "pairwise_disagreement.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["teacher_a", "teacher_b", "mean_abs_log_diff", "clips"]
        )
        writer.writeheader()
        writer.writerows(pairwise_rows)

    with (output_dir / "teacher_priors.json").open("w") as f:
        json.dump(teacher_priors, f, indent=2)

    print(
        json.dumps(
            {"output_dir": str(output_dir), "teacher_priors": teacher_priors}, indent=2
        )
    )


if __name__ == "__main__":
    main()
