"""Quick clean-vs-degraded validation sweep for original VDA checkpoints.

Evaluates Video Depth Anything small and large checkpoints on the deterministic
TartanAir validation split, reporting clean metrics plus a configurable set of
degradation levels built from the repo's native low-light pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.degradation import LowLightDegradation
from dataloader.tartanair_v2 import TartanAirV2Dataset
from evaluation.metrics.temporal_metrics import frame_depth_variation
from models.video_depth_anything_model import VideoDepthAnythingModel


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    lux_level: float | None
    severity_scale: float

    @property
    def is_clean(self) -> bool:
        return self.lux_level is None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    variant: str
    checkpoint_path: Path


class DepthMetricAccumulator:
    def __init__(self, *, min_depth: float = 1e-3, max_depth: float = 80.0) -> None:
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.valid_pixels = 0
        self.sum_abs_rel = 0.0
        self.sum_sq_rel = 0.0
        self.sum_sq_error = 0.0
        self.sum_sq_log_error = 0.0
        self.sum_log_diff = 0.0
        self.delta1_hits = 0
        self.delta2_hits = 0
        self.delta3_hits = 0

    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        pred_flat = pred.float().flatten()
        gt_flat = gt.float().flatten()
        if mask is not None:
            valid = mask.flatten().bool()
        else:
            valid = torch.ones_like(gt_flat, dtype=torch.bool)

        valid = valid & (gt_flat > self.min_depth) & (gt_flat < self.max_depth)
        valid = valid & (pred_flat > self.min_depth)
        valid = valid & torch.isfinite(pred_flat) & torch.isfinite(gt_flat)
        if not valid.any():
            return

        pred_valid = pred_flat[valid]
        gt_valid = gt_flat[valid]
        diff = pred_valid - gt_valid
        abs_diff = diff.abs()
        log_diff = torch.log(pred_valid) - torch.log(gt_valid)
        ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)

        count = int(pred_valid.numel())
        self.valid_pixels += count
        self.sum_abs_rel += float((abs_diff / gt_valid).sum().item())
        self.sum_sq_rel += float((((diff**2) / gt_valid).sum()).item())
        self.sum_sq_error += float(((diff**2).sum()).item())
        self.sum_sq_log_error += float(((log_diff**2).sum()).item())
        self.sum_log_diff += float(log_diff.sum().item())
        self.delta1_hits += int((ratio < 1.25).sum().item())
        self.delta2_hits += int((ratio < 1.25**2).sum().item())
        self.delta3_hits += int((ratio < 1.25**3).sum().item())

    def summary(self) -> dict[str, float | int]:
        if self.valid_pixels == 0:
            return {
                "valid_pixels": 0,
                "abs_rel": float("nan"),
                "sq_rel": float("nan"),
                "rmse": float("nan"),
                "rmse_log": float("nan"),
                "si_log": float("nan"),
                "delta1": float("nan"),
                "delta2": float("nan"),
                "delta3": float("nan"),
            }

        denom = float(self.valid_pixels)
        mean_sq_error = self.sum_sq_error / denom
        mean_sq_log_error = self.sum_sq_log_error / denom
        mean_log_diff = self.sum_log_diff / denom
        si_log = max(0.0, mean_sq_log_error - (mean_log_diff**2))
        return {
            "valid_pixels": self.valid_pixels,
            "abs_rel": self.sum_abs_rel / denom,
            "sq_rel": self.sum_sq_rel / denom,
            "rmse": mean_sq_error**0.5,
            "rmse_log": mean_sq_log_error**0.5,
            "si_log": si_log,
            "delta1": self.delta1_hits / denom,
            "delta2": self.delta2_hits / denom,
            "delta3": self.delta3_hits / denom,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/tartanair_v2.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--small-checkpoint", default="checkpoints/metric_video_depth_anything_vits.pth"
    )
    parser.add_argument(
        "--large-checkpoint", default="checkpoints/metric_video_depth_anything_vitl.pth"
    )
    parser.add_argument(
        "--models",
        default="small,large",
        help="Comma-separated subset of models to evaluate: small, large, or small,large",
    )
    parser.add_argument("--num-clips", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument(
        "--inference-size",
        default=None,
        help="Optional H,W override for VDA inference resolution, e.g. 518,518.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp", choices=("auto", "none", "fp16", "bf16"), default="auto"
    )
    parser.add_argument(
        "--mode", choices=("offline", "streaming_emulated"), default="offline"
    )
    parser.add_argument(
        "--strict-checkpoint",
        action="store_true",
        help="Require the checkpoint to load without missing or mismatched tensors.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip the clean validation condition and evaluate only the requested degraded condition(s).",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Evaluate only the clean validation condition.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing summary.json in the output directory if present.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N batches within a condition. Set to 0 to disable.",
    )
    parser.add_argument(
        "--degradation-level",
        action="append",
        dest="degradation_levels",
        default=None,
        help="Repeated spec formatted as name,lux[,severity]. Example: mild,20,0.5",
    )
    return parser.parse_args()


def _default_degradation_specs() -> list[ConditionSpec]:
    return [
        ConditionSpec(name="mild", lux_level=20.0, severity_scale=0.50),
        ConditionSpec(name="medium", lux_level=10.0, severity_scale=0.75),
        ConditionSpec(name="hard", lux_level=5.0, severity_scale=1.00),
    ]


def _parse_degradation_specs(raw_specs: Iterable[str] | None) -> list[ConditionSpec]:
    if not raw_specs:
        return _default_degradation_specs()

    parsed: list[ConditionSpec] = []
    for raw_spec in raw_specs:
        parts = [part.strip() for part in raw_spec.split(",") if part.strip()]
        if len(parts) not in (2, 3):
            raise ValueError(
                "Each --degradation-level must be formatted as name,lux or name,lux,severity. "
                f"Got: {raw_spec!r}"
            )
        name = parts[0]
        lux_level = float(parts[1])
        severity_scale = float(parts[2]) if len(parts) == 3 else 1.0
        parsed.append(
            ConditionSpec(name=name, lux_level=lux_level, severity_scale=severity_scale)
        )
    return parsed


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_models(raw_models: str) -> list[str]:
    requested = [part.strip().lower() for part in raw_models.split(",") if part.strip()]
    if not requested:
        raise ValueError("At least one model must be specified via --models")

    valid = {"small", "large"}
    unknown = [name for name in requested if name not in valid]
    if unknown:
        raise ValueError(
            f"Unknown model(s) requested via --models: {', '.join(sorted(set(unknown)))}"
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for name in requested:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _autocast_context(device: torch.device, amp_mode: str):
    if device.type != "cuda":
        return nullcontext()
    if amp_mode == "none":
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
        raise ValueError(
            f"--inference-size must use positive integers. Got: {raw_hw!r}"
        )
    return (height, width)


def _build_degradation(
    cfg: dict, condition: ConditionSpec
) -> LowLightDegradation | None:
    if condition.is_clean:
        return None

    degradation_cfg = cfg.get("degradation", {}) or {}
    degradation = LowLightDegradation(
        lux_level=float(condition.lux_level),
        poisson_scale=float(degradation_cfg.get("poisson_scale", 0.02)),
        gaussian_std=float(degradation_cfg.get("gaussian_std", 0.01)),
        blur_kernel_size=int(degradation_cfg.get("blur_kernel_size", 15)),
        blur_intensity=float(degradation_cfg.get("blur_intensity", 1.0)),
        temporal_variance=float(degradation_cfg.get("temporal_variance", 0.05)),
        apply_brightness_drop=bool(degradation_cfg.get("apply_brightness_drop", True)),
    )
    degradation.set_severity_scale(condition.severity_scale)
    return degradation


def _build_dataset(
    cfg: dict,
    *,
    condition: ConditionSpec,
    num_frames: int,
    seed: int,
    num_clips: int,
) -> TartanAirV2Dataset | Subset:
    data_cfg = cfg.get("data", {}) or {}
    degradation = _build_degradation(cfg, condition)
    dataset = TartanAirV2Dataset(
        root=str(data_cfg["root"]),
        num_frames=num_frames,
        image_size=tuple(int(v) for v in data_cfg.get("image_size", (256, 256))),
        max_trajectories=data_cfg.get(
            "max_val_trajectories", data_cfg.get("max_trajectories")
        ),
        clip_stride=int(data_cfg.get("clip_stride", 8)),
        frame_stride=int(data_cfg.get("frame_stride", 1)),
        split="val",
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=seed,
        difficulty=str(data_cfg.get("difficulty", "Data_easy")),
        camera=str(data_cfg.get("camera", "lcam_front")),
        max_depth=float(data_cfg.get("max_depth", 80.0)),
        envs=data_cfg.get("envs"),
        teacher_cache_dir=None,
        degradation=degradation,
        return_clean_and_degraded=degradation is not None,
        return_clean_reference=False,
    )
    if num_clips > 0 and len(dataset) > num_clips:
        return Subset(dataset, list(range(num_clips)))
    return dataset


def _build_loader(
    dataset: TartanAirV2Dataset | Subset, batch_size: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def _build_model(
    model_spec: ModelSpec,
    mode: str,
    eval_num_frames: int,
    device: torch.device,
    *,
    strict_checkpoint: bool,
) -> VideoDepthAnythingModel:
    if not model_spec.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_spec.checkpoint_path}")

    model_num_frames = max(32, int(eval_num_frames))
    model = VideoDepthAnythingModel(
        variant=model_spec.variant,
        num_frames=model_num_frames,
        checkpoint_path=str(model_spec.checkpoint_path),
        strict_checkpoint=strict_checkpoint,
        mode=mode,
    )
    return model.to(device).eval()


def _resize_video_tensor(
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
    kwargs: dict[str, object] = {"size": size, "mode": mode}
    if align_corners is not None:
        kwargs["align_corners"] = align_corners
    resized = torch.nn.functional.interpolate(flat, **kwargs)
    return (
        resized.reshape(batch_size, num_frames, channels, size[0], size[1])
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


@torch.inference_mode()
def _evaluate_condition(
    model: VideoDepthAnythingModel,
    loader: DataLoader,
    *,
    model_name: str,
    condition: ConditionSpec,
    device: torch.device,
    amp_mode: str,
    max_depth: float,
    mode: str,
    inference_size: tuple[int, int] | None = None,
    log_every: int = 0,
) -> dict[str, float | int | str | None]:
    metrics = DepthMetricAccumulator(max_depth=max_depth)
    total_clips = 0
    total_frames = 0
    temporal_sum = 0.0
    wall_start = time.perf_counter()
    dataset_len = len(loader.dataset)

    for batch_idx, batch in enumerate(loader, start=1):
        frame_key = "frames" if condition.is_clean else "degraded_frames"
        frames = batch[frame_key].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        original_depth_size = tuple(int(v) for v in depth.shape[-2:])
        model_frames = frames
        if (
            inference_size is not None
            and tuple(int(v) for v in frames.shape[-2:]) != inference_size
        ):
            model_frames = _resize_video_tensor(
                frames,
                size=inference_size,
                mode="bilinear",
                align_corners=False,
            )

        with _autocast_context(device, amp_mode):
            outputs = model(model_frames, mode=mode)
            pred = outputs["depth"]
            if (
                inference_size is not None
                and tuple(int(v) for v in pred.shape[-2:]) != original_depth_size
            ):
                pred = _resize_video_tensor(
                    pred,
                    size=original_depth_size,
                    mode="bilinear",
                    align_corners=False,
                )

        metrics.update(pred, depth, mask)
        temporal_sum += float(frame_depth_variation(pred, mask=mask.float())) * int(
            frames.shape[0]
        )
        total_clips += int(frames.shape[0])
        total_frames += int(frames.shape[0] * frames.shape[2])

        if log_every > 0 and (batch_idx % log_every == 0 or total_clips >= dataset_len):
            elapsed = max(time.perf_counter() - wall_start, 1e-6)
            print(
                f"[progress] model={model_name} condition={condition.name} "
                f"clips={total_clips}/{dataset_len} frames={total_frames} "
                f"clips_per_sec={total_clips / elapsed:.3f}",
                flush=True,
            )

    elapsed = time.perf_counter() - wall_start
    summary = metrics.summary()
    summary.update(
        {
            "condition": condition.name,
            "lux_level": condition.lux_level,
            "severity_scale": condition.severity_scale,
            "clips": total_clips,
            "frames": total_frames,
            "fdv": (temporal_sum / total_clips) if total_clips > 0 else float("nan"),
            "wall_time_sec": elapsed,
            "clips_per_sec": (total_clips / elapsed) if elapsed > 0 else float("nan"),
            "frames_per_sec": (total_frames / elapsed) if elapsed > 0 else float("nan"),
        }
    )
    return summary


def _write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "variant",
        "checkpoint",
        "inference_size",
        "condition",
        "lux_level",
        "severity_scale",
        "clips",
        "frames",
        "valid_pixels",
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "si_log",
        "delta1",
        "delta2",
        "delta3",
        "fdv",
        "wall_time_sec",
        "clips_per_sec",
        "frames_per_sec",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_dir: Path, report: dict[str, object], csv_rows: list[dict[str, object]]
) -> None:
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    with json_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    _write_summary_csv(csv_path, csv_rows)


def _csv_rows_from_report(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    results = report.get("results", {})
    if not isinstance(results, dict):
        return rows
    inference_size = report.get("inference_size")
    inference_size_label = None
    if isinstance(inference_size, list) and len(inference_size) == 2:
        inference_size_label = f"{inference_size[0]}x{inference_size[1]}"
    for model_name, model_payload in results.items():
        if not isinstance(model_payload, dict):
            continue
        variant = model_payload.get("variant")
        checkpoint = model_payload.get("checkpoint")
        conditions = model_payload.get("conditions", {})
        if not isinstance(conditions, dict):
            continue
        for summary in conditions.values():
            if not isinstance(summary, dict):
                continue
            rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "checkpoint": checkpoint,
                    "inference_size": inference_size_label,
                    **summary,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.clean_only and args.skip_clean:
        raise ValueError("--clean-only and --skip-clean cannot be used together")

    with Path(args.config).open("r") as handle:
        cfg = yaml.safe_load(handle) or {}

    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    data_cfg = cfg.get("data", {}) or {}
    eval_num_frames = int(
        args.num_frames or data_cfg.get("val_num_frames", data_cfg.get("num_frames", 8))
    )
    inference_size = _parse_hw(args.inference_size)
    max_depth = float(data_cfg.get("max_depth", 80.0))
    conditions: list[ConditionSpec] = []
    if args.clean_only:
        conditions.append(
            ConditionSpec(name="clean", lux_level=None, severity_scale=0.0)
        )
    else:
        if not args.skip_clean:
            conditions.append(
                ConditionSpec(name="clean", lux_level=None, severity_scale=0.0)
            )
        conditions.extend(_parse_degradation_specs(args.degradation_levels))
    requested_models = _parse_models(args.models)

    models: list[ModelSpec] = []
    if "small" in requested_models:
        models.append(
            ModelSpec(
                name="vda_small",
                variant="small",
                checkpoint_path=Path(args.small_checkpoint),
            )
        )
    if "large" in requested_models:
        models.append(
            ModelSpec(
                name="vda_large",
                variant="large",
                checkpoint_path=Path(args.large_checkpoint),
            )
        )

    if args.resume and json_path.is_file():
        report = json.loads(json_path.read_text())
        if not isinstance(report, dict):
            report = {}
    else:
        report = {}
    report.update(
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": str(Path(args.config)),
            "device": str(device),
            "num_frames": eval_num_frames,
            "num_clips": args.num_clips,
            "inference_size": None if inference_size is None else list(inference_size),
            "mode": args.mode,
            "amp": args.amp,
            "strict_checkpoint": bool(args.strict_checkpoint),
            "skip_clean": bool(args.skip_clean),
            "clean_only": bool(args.clean_only),
            "resume": bool(args.resume),
            "requested_models": requested_models,
            "results": report.get("results", {})
            if isinstance(report.get("results", {}), dict)
            else {},
        }
    )
    csv_rows = _csv_rows_from_report(report)

    for model_spec in models:
        print(
            f"[start] model={model_spec.name} variant={model_spec.variant} "
            f"mode={args.mode} device={device} inference_size={inference_size}",
            flush=True,
        )
        model = _build_model(
            model_spec,
            args.mode,
            eval_num_frames,
            device,
            strict_checkpoint=bool(args.strict_checkpoint),
        )
        existing_model_results = report["results"].get(model_spec.name, {})
        if not isinstance(existing_model_results, dict):
            existing_model_results = {}
        model_results: dict[str, object] = {
            "variant": model_spec.variant,
            "checkpoint": str(model_spec.checkpoint_path),
            "conditions": existing_model_results.get("conditions", {})
            if isinstance(existing_model_results.get("conditions", {}), dict)
            else {},
        }

        for condition in conditions:
            if args.resume and condition.name in model_results["conditions"]:
                print(
                    f"[resume] skipping model={model_spec.name} condition={condition.name} "
                    f"because it already exists in {json_path}",
                    flush=True,
                )
                continue
            dataset = _build_dataset(
                cfg,
                condition=condition,
                num_frames=eval_num_frames,
                seed=args.seed,
                num_clips=args.num_clips,
            )
            print(
                f"[condition] model={model_spec.name} condition={condition.name} clips={len(dataset)} "
                f"lux={condition.lux_level} severity={condition.severity_scale}",
                flush=True,
            )
            loader = _build_loader(
                dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )
            summary = _evaluate_condition(
                model,
                loader,
                model_name=model_spec.name,
                condition=condition,
                device=device,
                amp_mode=args.amp,
                max_depth=max_depth,
                mode=args.mode,
                inference_size=inference_size,
                log_every=args.log_every,
            )
            model_results["conditions"][condition.name] = summary
            csv_rows.append(
                {
                    "model": model_spec.name,
                    "variant": model_spec.variant,
                    "checkpoint": str(model_spec.checkpoint_path),
                    "inference_size": None
                    if inference_size is None
                    else f"{inference_size[0]}x{inference_size[1]}",
                    **summary,
                }
            )
            print(
                f"[done] model={model_spec.name} condition={condition.name} "
                f"abs_rel={summary['abs_rel']:.5f} delta1={summary['delta1']:.4f} "
                f"rmse={summary['rmse']:.4f} wall_time_sec={summary['wall_time_sec']:.1f}",
                flush=True,
            )
            report["results"][model_spec.name] = model_results
            report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            csv_rows = _csv_rows_from_report(report)
            _write_outputs(output_dir, report, csv_rows)

        report["results"][model_spec.name] = model_results

    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    csv_rows = _csv_rows_from_report(report)
    _write_outputs(output_dir, report, csv_rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": str(json_path),
                "summary_csv": str(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
