#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the current TartanAir val split.

Writes:
- per-clip CSV
- aggregate JSON
- bootstrap CI JSON
- optional paired-bootstrap CI JSON against a reference per-clip CSV

By default this uses the full current validation split even if the training
config used capped validation trajectories, because that is the scientifically
safer setting for matched comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.degradation import build_degradation
from dataloader.tartanair_v2 import TartanAirV2Dataset
from evaluation.metrics.depth_metrics import compute_depth_metrics
from evaluation.metrics.temporal_metrics import frame_depth_variation
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config or config snapshot.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--mode", default=None, help="Optional forward mode override.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--amp",
        choices=("auto", "none", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument(
        "--input-source",
        choices=("auto", "degraded", "clean"),
        default="auto",
        help="Which clip tensor to evaluate when both clean and degraded frames are available.",
    )
    parser.add_argument(
        "--inference-size",
        default=None,
        help="Optional H,W override for model inference resolution.",
    )
    parser.add_argument(
        "--clip-limit",
        type=int,
        default=None,
        help="Optional clip cap for debugging only.",
    )
    parser.add_argument(
        "--clip-shard-index",
        type=int,
        default=0,
        help="0-based shard index over the validation clips. Use with --clip-shard-count > 1.",
    )
    parser.add_argument(
        "--clip-shard-count",
        type=int,
        default=1,
        help="Number of deterministic clip shards to split the validation set into.",
    )
    parser.add_argument(
        "--respect-config-val-budget",
        action="store_true",
        help="Use any max_val_trajectories/max_trajectories cap from the config instead of full val.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing per-clip CSV in the output directory if present.",
    )
    parser.add_argument(
        "--strict-checkpoint",
        action="store_true",
        help="Require strict checkpoint loading.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for clip-mean CIs. Set to 0 to disable.",
    )
    parser.add_argument(
        "--reference-per-clip-csv",
        default=None,
        help="Optional baseline per-clip CSV for paired bootstrap deltas on shared clips.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N batches. Set to 0 to disable.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Rewrite aggregate/progress artifacts every N processed batches. Set to 1 for maximal safety.",
    )
    return parser.parse_args()


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context(device: torch.device, amp_mode: str):
    if device.type != "cuda" or amp_mode == "none":
        return nullcontext()
    if amp_mode in ("auto", "fp16"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _parse_hw(raw_hw: str | None) -> tuple[int, int] | None:
    if raw_hw is None:
        return None
    parts = [part.strip() for part in raw_hw.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"--inference-size must be formatted as H,W. Got: {raw_hw!r}")
    height, width = (int(parts[0]), int(parts[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"--inference-size must use positive integers. Got: {raw_hw!r}")
    return (height, width)


def _resize_video_tensor(
    tensor: torch.Tensor,
    *,
    size: tuple[int, int],
    mode: str,
    align_corners: bool | None,
) -> torch.Tensor:
    batch_size, channels, num_frames, height, width = tensor.shape
    flat = tensor.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
    kwargs: dict[str, Any] = {"size": size, "mode": mode}
    if align_corners is not None:
        kwargs["align_corners"] = align_corners
    resized = torch.nn.functional.interpolate(flat, **kwargs)
    return resized.reshape(batch_size, num_frames, channels, size[0], size[1]).permute(0, 2, 1, 3, 4).contiguous()


def _build_val_dataset(
    cfg: dict,
    *,
    respect_config_val_budget: bool,
    clip_limit: int | None,
    clip_shard_index: int,
    clip_shard_count: int,
    seed_override: int | None,
    num_frames_override: int | None,
) -> TartanAirV2Dataset | Subset:
    data_cfg = deepcopy(cfg.get("data", {}) or {})
    if not respect_config_val_budget:
        data_cfg["max_val_trajectories"] = None
        data_cfg["max_trajectories"] = None

    eval_num_frames = int(num_frames_override or data_cfg.get("val_num_frames", data_cfg.get("num_frames", 8)))
    degradation = build_degradation(cfg)
    dataset = TartanAirV2Dataset(
        root=str(data_cfg["root"]),
        num_frames=eval_num_frames,
        image_size=tuple(int(v) for v in data_cfg.get("image_size", (256, 256))),
        max_trajectories=data_cfg.get("max_val_trajectories", data_cfg.get("max_trajectories")),
        clip_stride=int(data_cfg.get("clip_stride", 8)),
        frame_stride=int(data_cfg.get("frame_stride", 1)),
        split="val",
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=int(seed_override if seed_override is not None else data_cfg.get("seed", 42)),
        difficulty=str(data_cfg.get("difficulty", "Data_easy")),
        camera=str(data_cfg.get("camera", "lcam_front")),
        max_depth=float(data_cfg.get("max_depth", 80.0)),
        envs=data_cfg.get("envs"),
        teacher_cache_dir=None,
        degradation=degradation,
        return_clean_and_degraded=degradation is not None,
        return_clean_reference=True,
    )
    indices = list(range(len(dataset)))
    if clip_limit is not None and clip_limit > 0 and len(indices) > clip_limit:
        indices = indices[:clip_limit]

    shard_count = max(int(clip_shard_count), 1)
    shard_index = int(clip_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"--clip-shard-index must be in [0, {shard_count - 1}] when --clip-shard-count={shard_count}, "
            f"got {shard_index}"
        )
    if shard_count > 1:
        indices = [index for index in indices if (index % shard_count) == shard_index]

    if len(indices) == len(dataset):
        return dataset
    return Subset(dataset, indices)


def _build_loader(dataset: TartanAirV2Dataset | Subset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, strict: bool) -> None:
    if hasattr(model, "load_checkpoint"):
        model.load_checkpoint(checkpoint_path, strict=strict)  # type: ignore[attr-defined]
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "Strict checkpoint loading failed. "
            f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
        )


def _select_frames(batch: dict[str, Any], input_source: str) -> tuple[torch.Tensor, str]:
    if input_source == "degraded":
        if "degraded_frames" not in batch:
            raise KeyError("Requested --input-source=degraded but batch has no degraded_frames")
        return batch["degraded_frames"], "degraded_frames"
    if input_source == "clean":
        if "clean_frames" in batch:
            return batch["clean_frames"], "clean_frames"
        if "frames" in batch:
            return batch["frames"], "frames"
        raise KeyError("Requested --input-source=clean but batch has no clean/frames tensor")

    if "degraded_frames" in batch:
        return batch["degraded_frames"], "degraded_frames"
    if "clean_frames" in batch:
        return batch["clean_frames"], "clean_frames"
    if "frames" in batch:
        return batch["frames"], "frames"
    raise KeyError("Batch does not contain frames, clean_frames, or degraded_frames")


def _stringify_video_ids(video_id: Any) -> list[str]:
    if isinstance(video_id, list):
        return [str(v) for v in video_id]
    return [str(video_id)]


def _start_frames_to_list(start_frame: Any) -> list[int]:
    if torch.is_tensor(start_frame):
        return [int(v) for v in start_frame.cpu().tolist()]
    if isinstance(start_frame, list):
        return [int(v) for v in start_frame]
    return [int(start_frame)]


def _summarize_gate_stats_per_clip(gate_stats: torch.Tensor) -> list[dict[str, float]]:
    if gate_stats.ndim == 2:
        gate_stats = gate_stats.unsqueeze(0)
    if gate_stats.ndim != 3:
        return [{} for _ in range(int(gate_stats.shape[0]))]
    per_clip = []
    for clip_gate in gate_stats.float():
        stage_means = clip_gate.mean(dim=0)
        per_clip.append({f"gate_stage_{idx}": float(value.item()) for idx, value in enumerate(stage_means)})
    return per_clip


def _bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[metric_name]) for row in rows if row.get(metric_name) is not None and np.isfinite(float(row[metric_name]))],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"metric": metric_name, "n_clips": 0, "mean": None, "ci_low": None, "ci_high": None}
    if samples <= 0:
        return {
            "metric": metric_name,
            "n_clips": int(values.size),
            "mean": float(values.mean()),
            "ci_low": None,
            "ci_high": None,
        }
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    return {
        "metric": metric_name,
        "n_clips": int(values.size),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def _load_reference_rows(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            key = (str(row["video_id"]), int(row["start_frame"]))
            rows[key] = {
                "abs_rel": float(row["abs_rel"]),
                "delta1": float(row["delta1"]),
            }
    return rows


def _paired_bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    reference_rows: dict[tuple[str, int], dict[str, float]],
    metric_name: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    paired_deltas = []
    for row in rows:
        key = (str(row["video_id"]), int(row["start_frame"]))
        ref = reference_rows.get(key)
        if ref is None:
            continue
        current_value = float(row[metric_name])
        reference_value = float(ref[metric_name])
        if not (np.isfinite(current_value) and np.isfinite(reference_value)):
            continue
        paired_deltas.append(current_value - reference_value)

    values = np.asarray(paired_deltas, dtype=np.float64)
    if values.size == 0:
        return {
            "metric": metric_name,
            "n_shared_clips": 0,
            "mean_delta_current_minus_reference": None,
            "ci_low": None,
            "ci_high": None,
        }
    if samples <= 0:
        return {
            "metric": metric_name,
            "n_shared_clips": int(values.size),
            "mean_delta_current_minus_reference": float(values.mean()),
            "ci_low": None,
            "ci_high": None,
        }
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    return {
        "metric": metric_name,
        "n_shared_clips": int(values.size),
        "mean_delta_current_minus_reference": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def _per_clip_fieldnames() -> list[str]:
    return [
        "clip_index",
        "video_id",
        "start_frame",
        "input_key",
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "si_log",
        "delta1",
        "delta2",
        "delta3",
        "fdv",
        "gate_stage_0",
        "gate_stage_1",
        "gate_stage_2",
        "gate_stage_3",
    ]


def _load_existing_per_clip_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, Any] = {
                "clip_index": int(raw_row["clip_index"]),
                "video_id": str(raw_row["video_id"]),
                "start_frame": int(raw_row["start_frame"]),
                "input_key": str(raw_row["input_key"]),
            }
            for key in ("abs_rel", "sq_rel", "rmse", "rmse_log", "si_log", "delta1", "delta2", "delta3", "fdv"):
                row[key] = float(raw_row[key])
            for key in ("gate_stage_0", "gate_stage_1", "gate_stage_2", "gate_stage_3"):
                value = raw_row.get(key, "")
                row[key] = float(value) if value not in ("", None) else None
            rows.append(row)
    return rows


def _rows_to_seen_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["video_id"]), int(row["start_frame"])) for row in rows}


def _aggregate_from_rows(rows: list[dict[str, Any]], *, wall_time_sec: float) -> dict[str, Any]:
    metric_names = ("abs_rel", "sq_rel", "rmse", "rmse_log", "si_log", "delta1", "delta2", "delta3", "fdv")
    aggregate = {
        "clips": len(rows),
        "wall_time_sec": float(wall_time_sec),
        "clips_per_sec": (len(rows) / wall_time_sec) if wall_time_sec > 0 else float("nan"),
    }
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in rows
            if row.get(metric_name) is not None and np.isfinite(float(row[metric_name]))
        ]
        aggregate[metric_name] = float(np.mean(values)) if values else float("nan")
    return aggregate


def _write_per_clip_csv_header(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_per_clip_fieldnames())
        writer.writeheader()


def _append_per_clip_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_per_clip_fieldnames())
        writer.writerows(rows)
        handle.flush()


def _write_progress_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    artifacts: dict[str, Any],
    device: torch.device,
    amp_mode: str,
    mode: str | None,
    input_source: str,
    inference_size: tuple[int, int] | None,
    log_every: int,
    flush_every: int,
    max_depth: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    per_clip_csv = artifacts["per_clip_csv"]
    aggregate_json = artifacts["aggregate_json"]
    progress_json = artifacts["progress_json"]
    per_clip_rows = list(artifacts["rows"])
    seen_keys = set(artifacts["seen_keys"])
    resumed_rows = int(artifacts["resumed_rows"])
    previous_wall_time_sec = float(artifacts["previous_wall_time_sec"])
    processed_batches = 0
    skipped_batches = 0
    wall_start = time.perf_counter()
    total_batches = len(loader)
    flush_every = max(int(flush_every), 1)

    def flush_progress(*, complete: bool) -> dict[str, Any]:
        elapsed = previous_wall_time_sec + max(time.perf_counter() - wall_start, 0.0)
        aggregate = _aggregate_from_rows(per_clip_rows, wall_time_sec=elapsed)
        progress_payload = {
            "complete": bool(complete),
            "clips_evaluated": len(per_clip_rows),
            "resumed_rows": resumed_rows,
            "processed_batches": processed_batches,
            "skipped_batches": skipped_batches,
            "total_batches": total_batches,
            "elapsed_wall_time_sec": elapsed,
        }
        partial_report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "complete": bool(complete),
            "aggregate": aggregate,
            "progress": progress_payload,
            "outputs": {
                "per_clip_csv": str(per_clip_csv),
                "aggregate_json": str(aggregate_json),
            },
        }
        aggregate_json.write_text(json.dumps(partial_report, indent=2, sort_keys=True) + "\n")
        _write_progress_json(progress_json, progress_payload)
        return aggregate

    for batch_idx, batch in enumerate(loader, start=1):
        batch_video_ids = _stringify_video_ids(batch["video_id"])
        batch_start_frames = _start_frames_to_list(batch["start_frame"])
        batch_keys = [(batch_video_ids[idx], batch_start_frames[idx]) for idx in range(len(batch_video_ids))]
        if batch_keys and all(key in seen_keys for key in batch_keys):
            skipped_batches += 1
            continue

        frames, frame_key = _select_frames(batch, input_source)
        frames = frames.to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        original_depth_size = tuple(int(v) for v in depth.shape[-2:])
        model_frames = frames
        if inference_size is not None and tuple(int(v) for v in frames.shape[-2:]) != inference_size:
            model_frames = _resize_video_tensor(
                frames,
                size=inference_size,
                mode="bilinear",
                align_corners=False,
            )

        model_kwargs: dict[str, Any] = {}
        if mode is not None:
            model_kwargs["mode"] = mode
        wants_gate_stats = bool(getattr(model, "state_gate_enabled", False))

        with _autocast_context(device, amp_mode):
            try:
                outputs = model(model_frames, return_gate_stats=wants_gate_stats, **model_kwargs)
            except TypeError:
                outputs = model(model_frames, **model_kwargs)
            pred = outputs["depth"]
            if inference_size is not None and tuple(int(v) for v in pred.shape[-2:]) != original_depth_size:
                pred = _resize_video_tensor(
                    pred,
                    size=original_depth_size,
                    mode="bilinear",
                    align_corners=False,
                )

        gate_rows = _summarize_gate_stats_per_clip(outputs["gate_stats"]) if "gate_stats" in outputs else [{}] * int(pred.shape[0])
        video_ids = batch_video_ids
        start_frames = batch_start_frames
        new_rows: list[dict[str, Any]] = []

        for sample_idx in range(int(pred.shape[0])):
            key = (video_ids[sample_idx], start_frames[sample_idx])
            if key in seen_keys:
                continue
            clip_pred = pred[sample_idx : sample_idx + 1]
            clip_depth = depth[sample_idx : sample_idx + 1]
            clip_mask = mask[sample_idx : sample_idx + 1]
            metrics = compute_depth_metrics(
                clip_pred.float(),
                clip_depth.float(),
                mask=clip_mask,
                max_depth=max_depth,
            )
            metrics["fdv"] = float(frame_depth_variation(clip_pred.float(), mask=clip_mask.float()))
            row = {
                "clip_index": len(per_clip_rows),
                "video_id": video_ids[sample_idx],
                "start_frame": start_frames[sample_idx],
                "input_key": frame_key,
                "abs_rel": float(metrics["abs_rel"]),
                "sq_rel": float(metrics["sq_rel"]),
                "rmse": float(metrics["rmse"]),
                "rmse_log": float(metrics["rmse_log"]),
                "si_log": float(metrics["si_log"]),
                "delta1": float(metrics["delta1"]),
                "delta2": float(metrics["delta2"]),
                "delta3": float(metrics["delta3"]),
                "fdv": float(metrics["fdv"]),
            }
            row.update(gate_rows[sample_idx])
            per_clip_rows.append(row)
            new_rows.append(row)
            seen_keys.add(key)

        if new_rows:
            _append_per_clip_rows(per_clip_csv, new_rows)
        processed_batches += 1

        if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == total_batches):
            elapsed = max(previous_wall_time_sec + (time.perf_counter() - wall_start), 1e-6)
            print(
                f"[progress] batches={batch_idx}/{total_batches} clips={len(per_clip_rows)} "
                f"clips_per_sec={len(per_clip_rows) / elapsed:.3f}",
                flush=True,
            )
        if processed_batches > 0 and processed_batches % flush_every == 0:
            flush_progress(complete=False)

    aggregate = flush_progress(complete=False)
    progress = {
        "resumed_rows": resumed_rows,
        "processed_batches": processed_batches,
        "skipped_batches": skipped_batches,
        "total_batches": total_batches,
    }
    return per_clip_rows, aggregate, progress


def _write_per_clip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_per_clip_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r") as handle:
        cfg = yaml.safe_load(handle) or {}

    cfg_for_model = deepcopy(cfg)
    cfg_for_model.setdefault("model", {})
    cfg_for_model["model"]["checkpoint_path"] = None
    cfg_for_model["model"]["strict_checkpoint"] = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_clip_csv = output_dir / "per_clip_metrics.csv"
    aggregate_json = output_dir / "aggregate_metrics.json"
    bootstrap_json = output_dir / "bootstrap_ci.json"
    paired_bootstrap_json = output_dir / "paired_bootstrap_ci.json"
    progress_json = output_dir / "progress.json"

    existing_rows = _load_existing_per_clip_rows(per_clip_csv) if args.resume else []
    if args.resume and existing_rows:
        print(f"[resume] loaded {len(existing_rows)} rows from {per_clip_csv}", flush=True)
    elif not existing_rows:
        _write_per_clip_csv_header(per_clip_csv)

    previous_wall_time_sec = 0.0
    if args.resume and aggregate_json.is_file():
        try:
            existing_report = json.loads(aggregate_json.read_text())
            previous_wall_time_sec = float(existing_report.get("aggregate", {}).get("wall_time_sec", 0.0) or 0.0)
        except Exception:
            previous_wall_time_sec = 0.0

    artifacts = {
        "per_clip_csv": per_clip_csv,
        "aggregate_json": aggregate_json,
        "progress_json": progress_json,
        "rows": existing_rows,
        "seen_keys": _rows_to_seen_keys(existing_rows),
        "resumed_rows": len(existing_rows),
        "previous_wall_time_sec": previous_wall_time_sec,
    }

    device = _resolve_device(args.device)
    dataset = _build_val_dataset(
        cfg,
        respect_config_val_budget=bool(args.respect_config_val_budget),
        clip_limit=args.clip_limit,
        clip_shard_index=int(args.clip_shard_index),
        clip_shard_count=int(args.clip_shard_count),
        seed_override=args.seed,
        num_frames_override=args.num_frames,
    )
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int((cfg.get("training", {}) or {}).get("val_num_workers", 4))
    )
    loader = _build_loader(dataset, batch_size=int(args.batch_size), num_workers=num_workers)

    model = build_model(cfg_for_model).to(device).eval()
    _load_checkpoint(model, args.checkpoint, strict=bool(args.strict_checkpoint))

    data_cfg = cfg.get("data", {}) or {}
    inference_size = _parse_hw(args.inference_size)
    eval_mode = args.mode if args.mode is not None else cfg.get("model", {}).get("mode", None)
    max_depth = float(data_cfg.get("max_depth", 80.0))

    per_clip_rows, aggregate, progress = evaluate(
        model,
        loader,
        artifacts=artifacts,
        device=device,
        amp_mode=args.amp,
        mode=eval_mode,
        input_source=args.input_source,
        inference_size=inference_size,
        log_every=int(args.log_every),
        flush_every=int(args.flush_every),
        max_depth=max_depth,
    )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "complete": True,
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(device),
        "mode": eval_mode,
        "input_source": args.input_source,
        "respect_config_val_budget": bool(args.respect_config_val_budget),
        "clip_limit": args.clip_limit,
        "clip_shard_index": int(args.clip_shard_index),
        "clip_shard_count": int(args.clip_shard_count),
        "resume": bool(args.resume),
        "batch_size": int(args.batch_size),
        "num_workers": num_workers,
        "bootstrap_samples": int(args.bootstrap_samples),
        "inference_size": None if inference_size is None else list(inference_size),
        "aggregate": aggregate,
        "progress": {
            **progress,
            "complete": True,
            "clips_evaluated": len(per_clip_rows),
            "elapsed_wall_time_sec": aggregate["wall_time_sec"],
        },
        "outputs": {
            "per_clip_csv": str(per_clip_csv),
            "aggregate_json": str(aggregate_json),
            "bootstrap_json": str(bootstrap_json),
        },
    }
    bootstrap_report = {
        metric_name: _bootstrap_ci(
            per_clip_rows,
            metric_name=metric_name,
            samples=int(args.bootstrap_samples),
            seed=12345 + idx,
        )
        for idx, metric_name in enumerate(("abs_rel", "delta1", "rmse", "si_log", "fdv"))
    }
    bootstrap_json.write_text(json.dumps(bootstrap_report, indent=2, sort_keys=True) + "\n")

    if args.reference_per_clip_csv:
        reference_rows = _load_reference_rows(Path(args.reference_per_clip_csv))
        paired_report = {
            metric_name: _paired_bootstrap_ci(
                per_clip_rows,
                reference_rows=reference_rows,
                metric_name=metric_name,
                samples=int(args.bootstrap_samples),
                seed=54321 + idx,
            )
            for idx, metric_name in enumerate(("abs_rel", "delta1"))
        }
        paired_bootstrap_json.write_text(json.dumps(paired_report, indent=2, sort_keys=True) + "\n")
        report["outputs"]["paired_bootstrap_json"] = str(paired_bootstrap_json)

    aggregate_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_progress_json(progress_json, report["progress"])

    print(
        f"[done] clips={aggregate['clips']} abs_rel={aggregate.get('abs_rel', float('nan')):.5f} "
        f"delta1={aggregate.get('delta1', float('nan')):.4f} per_clip_csv={per_clip_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
