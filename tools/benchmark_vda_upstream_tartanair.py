"""Benchmark upstream Video Depth Anything streaming inference on degraded TartanAir validation.

This driver uses the original repo's streaming model implementation directly:
`video_depth_anything.video_depth_stream.VideoDepthAnything`.

It exports one deterministic degraded validation split from raw TartanAir frames,
then benchmarks the metric checkpoints under two temporal-reset modes:

- trajectory_stream: reset only at trajectory boundaries
- clip_reset: reset at every validation clip boundary
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.degradation import LowLightParams, build_degradation
from dataloader.tartanair_depth import (
    decode_tartanair_depth as decode_tartanair_depth_png,
)

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}
DEFAULT_CHECKPOINTS = {
    "vits": PROJECT_ROOT / "checkpoints/metric_video_depth_anything_vits.pth",
    "vitl": PROJECT_ROOT / "checkpoints/metric_video_depth_anything_vitl.pth",
}
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/generated/focused_redirection_20260310/05_vdal_degraded_stream.yaml"
)
DEFAULT_UPSTREAM_REPO = Path("/projects/b5dh/repos/Video-Depth-Anything")
DEFAULT_EXPORT_CHUNK_SIZE = 64
DEFAULT_MAX_RES = 1280
UPSTREAM_SCOPES = ("trajectory_stream", "clip_reset")


@dataclass(frozen=True)
class TrajectoryRecord:
    video_id: str
    image_dir: Path
    depth_dir: Path
    image_files: tuple[Path, ...]
    depth_files: tuple[Path, ...]


class DepthMetricAccumulator:
    """Pixel-weighted metric accumulator matching the repo's paper metrics."""

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

    def update(self, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        pred = prediction.astype(np.float32, copy=False).reshape(-1)
        gt = ground_truth.astype(np.float32, copy=False).reshape(-1)
        valid = (
            np.isfinite(pred)
            & np.isfinite(gt)
            & (pred > self.min_depth)
            & (gt > self.min_depth)
            & (gt < self.max_depth)
        )
        if not np.any(valid):
            return

        pred_valid = pred[valid].astype(np.float64, copy=False)
        gt_valid = gt[valid].astype(np.float64, copy=False)
        diff = pred_valid - gt_valid
        abs_diff = np.abs(diff)
        log_diff = np.log(pred_valid) - np.log(gt_valid)
        ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)

        count = int(pred_valid.size)
        self.valid_pixels += count
        self.sum_abs_rel += float(np.sum(abs_diff / gt_valid))
        self.sum_sq_rel += float(np.sum((diff**2) / gt_valid))
        self.sum_sq_error += float(np.sum(diff**2))
        self.sum_sq_log_error += float(np.sum(log_diff**2))
        self.sum_log_diff += float(np.sum(log_diff))
        self.delta1_hits += int(np.sum(ratio < 1.25))
        self.delta2_hits += int(np.sum(ratio < 1.25**2))
        self.delta3_hits += int(np.sum(ratio < 1.25**3))

    def merge(self, other: "DepthMetricAccumulator") -> None:
        self.valid_pixels += other.valid_pixels
        self.sum_abs_rel += other.sum_abs_rel
        self.sum_sq_rel += other.sum_sq_rel
        self.sum_sq_error += other.sum_sq_error
        self.sum_sq_log_error += other.sum_sq_log_error
        self.sum_log_diff += other.sum_log_diff
        self.delta1_hits += other.delta1_hits
        self.delta2_hits += other.delta2_hits
        self.delta3_hits += other.delta3_hits

    def as_metrics(self) -> dict[str, float | int]:
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


def decode_tartanair_depth(path: Path) -> np.ndarray:
    """Decode a TartanAir depth PNG to float32 metres."""
    return decode_tartanair_depth_png(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", default="vits,vitl")
    parser.add_argument(
        "--stream-scope",
        choices=("trajectory_stream", "clip_reset", "both"),
        default="both",
    )
    parser.add_argument(
        "--metric-suite", choices=("upstream", "paper", "both"), default="both"
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite-export", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--debug-max-trajectories",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug-max-groups-per-mode",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def parse_model_names(raw_models: str) -> list[str]:
    requested = [part.strip().lower() for part in raw_models.split(",") if part.strip()]
    if not requested:
        raise ValueError("At least one model must be requested via --models")
    unknown = [name for name in requested if name not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(set(unknown)))}")
    deduped: list[str] = []
    seen: set[str] = set()
    for name in requested:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def parse_scopes(raw_scope: str) -> list[str]:
    if raw_scope == "both":
        return list(UPSTREAM_SCOPES)
    return [raw_scope]


def resolve_device(raw_device: str) -> str:
    if raw_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw_device


def ensure_prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if sys.path and sys.path[0] == path_str:
        return
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def configure_runtime_cache(output_root: Path) -> None:
    cache_root = output_root.resolve() / ".runtime_cache"
    xdg_cache = cache_root / "xdg"
    torchinductor_cache = cache_root / "torchinductor"
    triton_cache = cache_root / "triton"
    torch_home = cache_root / "torch_home"
    for path in (xdg_cache, torchinductor_cache, triton_cache, torch_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_cache))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("TORCH_HOME", str(torch_home))


def bootstrap_upstream_utils(repo_root: Path) -> None:
    utils_root = repo_root / "utils"
    if not utils_root.is_dir():
        raise FileNotFoundError(utils_root)

    sys.modules.pop("utils.util", None)
    sys.modules.pop("utils.dc_utils", None)
    sys.modules.pop("utils", None)

    package = types.ModuleType("utils")
    package.__path__ = [str(utils_root)]  # type: ignore[attr-defined]
    package.__package__ = "utils"
    sys.modules["utils"] = package

    for module_name in ("util", "dc_utils"):
        module_path = utils_root / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(
            f"utils.{module_name}", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load upstream utils module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"utils.{module_name}"] = module
        spec.loader.exec_module(module)
        setattr(package, module_name, module)


def load_upstream_model(
    repo_root: Path, encoder: str, checkpoint_path: Path, device: str
):
    ensure_prepend_sys_path(repo_root)
    bootstrap_upstream_utils(repo_root)
    from video_depth_anything.video_depth_stream import VideoDepthAnything

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = VideoDepthAnything(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def sanitize_group_path(group_name: str) -> Path:
    return Path(*[part for part in group_name.split("/") if part])


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_validation_trajectories(
    cfg: dict, debug_max_trajectories: int | None
) -> list[TrajectoryRecord]:
    data_cfg = cfg.get("data", {}) or {}
    root = Path(data_cfg["root"])
    difficulty = str(data_cfg.get("difficulty", "Data_easy"))
    camera = str(data_cfg.get("camera", "lcam_front"))
    envs = data_cfg.get("envs")
    seed = int(data_cfg.get("seed", 42))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    max_val_trajectories = data_cfg.get(
        "max_val_trajectories", data_cfg.get("max_trajectories", None)
    )

    env_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if envs is not None:
        allowed = set(envs)
        env_dirs = [path for path in env_dirs if path.name in allowed]

    all_trajectories: list[tuple[Path, Path, str]] = []
    for env_dir in env_dirs:
        diff_dir = env_dir / difficulty
        if not diff_dir.is_dir():
            continue
        for traj_dir in sorted(diff_dir.iterdir()):
            if not traj_dir.is_dir():
                continue
            image_dir = traj_dir / f"image_{camera}"
            depth_dir = traj_dir / f"depth_{camera}"
            if image_dir.is_dir() and depth_dir.is_dir():
                video_id = f"{env_dir.name}/{traj_dir.name}"
                all_trajectories.append((image_dir, depth_dir, video_id))

    rng = random.Random(seed)
    shuffled = all_trajectories[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    selected = shuffled[:n_val]
    if max_val_trajectories is not None:
        selected = selected[: int(max_val_trajectories)]
    if debug_max_trajectories is not None:
        selected = selected[: int(debug_max_trajectories)]

    clip_num_frames = int(data_cfg.get("val_num_frames", data_cfg.get("num_frames", 8)))
    frame_stride = int(data_cfg.get("frame_stride", 1))
    clip_len_in_frames = (clip_num_frames - 1) * frame_stride + 1

    records: list[TrajectoryRecord] = []
    for image_dir, depth_dir, video_id in selected:
        image_files = tuple(sorted(image_dir.glob("*.png")))
        depth_files = tuple(sorted(depth_dir.glob("*.png")))
        n = min(len(image_files), len(depth_files))
        if n < clip_len_in_frames:
            continue
        records.append(
            TrajectoryRecord(
                video_id=video_id,
                image_dir=image_dir,
                depth_dir=depth_dir,
                image_files=image_files[:n],
                depth_files=depth_files[:n],
            )
        )
    return records


def slice_params(params: LowLightParams, start: int, end: int) -> LowLightParams:
    return LowLightParams(
        exposure_ev=params.exposure_ev[:, start:end],
        shot_scale=params.shot_scale[:, start:end],
        read_noise=params.read_noise[:, start:end],
        quant_step=params.quant_step[:, start:end],
        band_noise=params.band_noise[:, start:end],
        band_direction=params.band_direction[:, start:end],
        blur_sigma1=params.blur_sigma1[:, start:end],
        blur_sigma2=params.blur_sigma2[:, start:end],
        blur_angle=params.blur_angle[:, start:end],
    )


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor.to(device)


def save_rgb_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Failed to save image to {path}")


def ensure_prediction_shape(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> np.ndarray:
    if prediction.shape == ground_truth.shape:
        return prediction
    return cv2.resize(
        prediction.astype(np.float32),
        (ground_truth.shape[1], ground_truth.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


@lru_cache(maxsize=4096)
def load_ground_truth_depth(depth_path: str, max_depth: float) -> np.ndarray:
    depth = decode_tartanair_depth(Path(depth_path)).astype(np.float32)
    return np.clip(depth, 0.0, float(max_depth))


def export_complete(manifest: dict[str, object], export_root: Path) -> bool:
    for group in manifest.get("trajectory_stream", []):
        frames = group.get("frames", [])
        for item in frames:
            image_path = export_root / str(item["image"])
            if not image_path.is_file():
                return False
    return True


def build_manifest(
    trajectories: list[TrajectoryRecord],
    cfg: dict,
    export_root: Path,
    export_device: str,
) -> dict[str, object]:
    data_cfg = cfg.get("data", {}) or {}
    clip_num_frames = int(data_cfg.get("val_num_frames", data_cfg.get("num_frames", 8)))
    clip_stride = int(data_cfg.get("clip_stride", 8))
    frame_stride = int(data_cfg.get("frame_stride", 1))
    clip_len_in_frames = (clip_num_frames - 1) * frame_stride + 1
    source_root = Path(data_cfg["root"])

    trajectory_groups: list[dict[str, object]] = []
    clip_groups: list[dict[str, object]] = []
    for record in trajectories:
        trajectory_frames: list[dict[str, object]] = []
        for frame_index, (image_path, depth_path) in enumerate(
            zip(record.image_files, record.depth_files)
        ):
            rel_image = Path("images") / record.video_id / image_path.name
            rel_source_image = image_path.relative_to(source_root)
            rel_depth = depth_path.relative_to(source_root)
            trajectory_frames.append(
                {
                    "frame_index": frame_index,
                    "image": str(rel_image),
                    "source_image": str(rel_source_image),
                    "gt_depth": str(rel_depth),
                }
            )

        trajectory_groups.append(
            {
                "group_name": record.video_id,
                "video_id": record.video_id,
                "start_frame": 0,
                "frames": trajectory_frames,
            }
        )

        n_frames = len(record.image_files)
        for start in range(0, n_frames - clip_len_in_frames + 1, clip_stride):
            frame_entries = []
            for offset in range(clip_num_frames):
                frame_index = start + offset * frame_stride
                frame_entries.append(trajectory_frames[frame_index])
            clip_groups.append(
                {
                    "group_name": f"{record.video_id}/clip_{start:06d}",
                    "video_id": record.video_id,
                    "start_frame": start,
                    "frames": frame_entries,
                }
            )

    return {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": str(cfg.get("__config_path__", "")),
            "source_root": str(source_root),
            "export_root": str(export_root),
            "export_device": export_device,
            "seed": int(data_cfg.get("seed", 42)),
            "difficulty": str(data_cfg.get("difficulty", "Data_easy")),
            "camera": str(data_cfg.get("camera", "lcam_front")),
            "val_fraction": float(data_cfg.get("val_fraction", 0.1)),
            "clip_num_frames": clip_num_frames,
            "clip_stride": clip_stride,
            "frame_stride": frame_stride,
            "max_depth": float(data_cfg.get("max_depth", 80.0)),
            "num_trajectories": len(trajectory_groups),
            "num_clip_groups": len(clip_groups),
            "degradation": cfg.get("degradation", {}),
        },
        "trajectory_stream": trajectory_groups,
        "clip_reset": clip_groups,
    }


def export_degraded_dataset(
    cfg: dict,
    export_root: Path,
    *,
    overwrite_export: bool,
    debug_max_trajectories: int | None,
) -> dict[str, object]:
    manifest_path = export_root / "manifest.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if manifest_path.is_file() and not overwrite_export:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if export_complete(manifest, export_root):
            print(
                f"[export] Reusing existing degraded export: {manifest_path}",
                flush=True,
            )
            return manifest
        print(
            f"[export] Existing manifest is incomplete; rebuilding: {manifest_path}",
            flush=True,
        )

    degradation = build_degradation(cfg)
    if degradation is None:
        raise ValueError("Benchmark export requires config.degradation.enabled=true")

    export_root.mkdir(parents=True, exist_ok=True)
    trajectories = discover_validation_trajectories(cfg, debug_max_trajectories)
    manifest = build_manifest(trajectories, cfg, export_root, device)
    images_root = export_root / "images"
    process_device = torch.device(device)

    for traj_idx, group in enumerate(manifest["trajectory_stream"], start=1):
        video_id = str(group["video_id"])
        frame_items = list(group["frames"])
        output_dir = images_root / video_id
        if (
            not overwrite_export
            and output_dir.is_dir()
            and len(list(output_dir.glob("*.png"))) == len(frame_items)
        ):
            print(
                f"[export] Reusing trajectory {traj_idx}/{len(manifest['trajectory_stream'])}: {video_id}",
                flush=True,
            )
            continue

        rng = np.random.RandomState(int(manifest["metadata"]["seed"]) + traj_idx - 1)
        generator = degradation._noise_generator(process_device, rng, None)
        params = degradation._build_params(
            batch_size=1,
            num_frames=len(frame_items),
            device=process_device,
            dtype=torch.float32,
            rng=rng,
            generator=None,
        )

        print(
            f"[export] Trajectory {traj_idx}/{len(manifest['trajectory_stream'])} {video_id} "
            f"frames={len(frame_items)} device={device}",
            flush=True,
        )
        for chunk_start in range(0, len(frame_items), DEFAULT_EXPORT_CHUNK_SIZE):
            chunk_end = min(chunk_start + DEFAULT_EXPORT_CHUNK_SIZE, len(frame_items))
            frame_tensors = []
            for item in frame_items[chunk_start:chunk_end]:
                source_image = Path(manifest["metadata"]["source_root"]) / str(
                    item["source_image"]
                )
                frame_tensors.append(load_rgb_tensor(source_image, process_device))
            clean_chunk = torch.stack(frame_tensors, dim=0)
            with torch.inference_mode():
                degraded_chunk = degradation(
                    clean_chunk,
                    params=slice_params(params, chunk_start, chunk_end),
                    generator=generator,
                )
            for local_idx, item in enumerate(frame_items[chunk_start:chunk_end]):
                output_path = export_root / str(item["image"])
                save_rgb_tensor(output_path, degraded_chunk[local_idx])

        if process_device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(manifest_path, manifest)
    print(
        f"[export] Completed degraded export: trajectories={manifest['metadata']['num_trajectories']} "
        f"clips={manifest['metadata']['num_clip_groups']} root={export_root}",
        flush=True,
    )
    return manifest


def load_degraded_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_frame_for_upstream(frame: np.ndarray, *, max_res: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if max_res > 0 and max(height, width) > max_res:
        scale = max_res / float(max(height, width))
        new_height = round(height * scale)
        new_width = round(width * scale)
        frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
    return frame


def prediction_path(
    output_root: Path, encoder: str, mode: str, group_name: str, frame_name: str
) -> Path:
    return (
        output_root
        / "predictions"
        / encoder
        / mode
        / sanitize_group_path(group_name)
        / Path(frame_name).with_suffix(".npy")
    )


def fieldnames_for_group_rows(include_upstream: bool, include_paper: bool) -> list[str]:
    fields = [
        "model",
        "checkpoint",
        "mode",
        "group_name",
        "video_id",
        "start_frame",
        "num_frames",
        "valid_pixels",
    ]
    if include_upstream:
        fields.extend(["upstream_absrel", "upstream_delta1"])
    if include_paper:
        fields.extend(
            [
                "paper_abs_rel",
                "paper_sq_rel",
                "paper_rmse",
                "paper_rmse_log",
                "paper_si_log",
                "paper_delta1",
                "paper_delta2",
                "paper_delta3",
            ]
        )
    return fields


def fieldnames_for_summary_rows(
    include_upstream: bool, include_paper: bool
) -> list[str]:
    fields = [
        "model",
        "checkpoint",
        "mode",
        "num_groups",
        "num_frames",
        "valid_pixels",
        "input_size",
        "fp32",
        "wall_time_sec",
    ]
    if include_upstream:
        fields.extend(["upstream_absrel", "upstream_delta1"])
    if include_paper:
        fields.extend(
            [
                "paper_abs_rel",
                "paper_sq_rel",
                "paper_rmse",
                "paper_rmse_log",
                "paper_si_log",
                "paper_delta1",
                "paper_delta2",
                "paper_delta3",
            ]
        )
    return fields


def metrics_to_row(
    metrics: dict[str, float | int], include_upstream: bool, include_paper: bool
) -> dict[str, object]:
    row: dict[str, object] = {"valid_pixels": metrics["valid_pixels"]}
    if include_upstream:
        row["upstream_absrel"] = metrics["abs_rel"]
        row["upstream_delta1"] = metrics["delta1"]
    if include_paper:
        row["paper_abs_rel"] = metrics["abs_rel"]
        row["paper_sq_rel"] = metrics["sq_rel"]
        row["paper_rmse"] = metrics["rmse"]
        row["paper_rmse_log"] = metrics["rmse_log"]
        row["paper_si_log"] = metrics["si_log"]
        row["paper_delta1"] = metrics["delta1"]
        row["paper_delta2"] = metrics["delta2"]
        row["paper_delta3"] = metrics["delta3"]
    return row


def run_scope_benchmark(
    *,
    args: argparse.Namespace,
    manifest: dict[str, object],
    model_name: str,
    checkpoint_path: Path,
    mode: str,
    device: str,
    include_upstream: bool,
    include_paper: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    groups = list(manifest[mode])
    if args.debug_max_groups_per_mode is not None:
        groups = groups[: int(args.debug_max_groups_per_mode)]
    model = load_upstream_model(args.repo_root, model_name, checkpoint_path, device)
    aggregate = DepthMetricAccumulator(
        max_depth=float(manifest["metadata"]["max_depth"])
    )
    group_rows: list[dict[str, object]] = []

    source_root = Path(manifest["metadata"]["source_root"])
    export_root = Path(manifest["metadata"]["export_root"])
    use_fp32 = bool(args.fp32 or device == "cpu")
    total_frames = 0
    wall_start = time.perf_counter()

    for group_idx, group in enumerate(groups, start=1):
        model.reset_stream_state()
        group_acc = DepthMetricAccumulator(
            max_depth=float(manifest["metadata"]["max_depth"])
        )
        frames = list(group["frames"])
        print(
            f"[benchmark] model={model_name} mode={mode} group={group_idx}/{len(groups)} "
            f"name={group['group_name']} frames={len(frames)}",
            flush=True,
        )
        for item in frames:
            image_path = export_root / str(item["image"])
            gt_path = source_root / str(item["gt_depth"])
            pred_path = prediction_path(
                args.output_root,
                model_name,
                mode,
                str(group["group_name"]),
                Path(str(item["image"])).name,
            )
            pred_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and pred_path.exists():
                prediction = np.load(pred_path).astype(np.float32)
            else:
                frame = load_degraded_rgb(image_path)
                frame = preprocess_frame_for_upstream(frame, max_res=DEFAULT_MAX_RES)
                prediction = model.infer_video_depth_one(
                    frame,
                    input_size=args.input_size,
                    device=device,
                    fp32=use_fp32,
                ).astype(np.float32)
                np.save(pred_path, prediction.astype(np.float32))

            ground_truth = load_ground_truth_depth(
                str(gt_path), float(manifest["metadata"]["max_depth"])
            )
            prediction = ensure_prediction_shape(prediction, ground_truth)
            group_acc.update(prediction, ground_truth)
            total_frames += 1

        group_metrics = group_acc.as_metrics()
        aggregate.merge(group_acc)
        group_row = {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "mode": mode,
            "group_name": group["group_name"],
            "video_id": group["video_id"],
            "start_frame": group.get("start_frame", 0),
            "num_frames": len(frames),
            **metrics_to_row(group_metrics, include_upstream, include_paper),
        }
        group_rows.append(group_row)

    summary_metrics = aggregate.as_metrics()
    summary_row = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "mode": mode,
        "num_groups": len(groups),
        "num_frames": total_frames,
        "valid_pixels": summary_metrics["valid_pixels"],
        "input_size": args.input_size,
        "fp32": use_fp32,
        **metrics_to_row(summary_metrics, include_upstream, include_paper),
        "wall_time_sec": time.perf_counter() - wall_start,
    }

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return summary_row, group_rows


def save_scope_reports(
    output_root: Path,
    mode: str,
    summary_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    *,
    include_upstream: bool,
    include_paper: bool,
    manifest: dict[str, object],
    args: argparse.Namespace,
) -> None:
    scope_root = output_root / "summary" / mode
    write_csv(
        scope_root / "summary.csv",
        summary_rows,
        fieldnames_for_summary_rows(include_upstream, include_paper),
    )
    write_csv(
        scope_root / "group_metrics.csv",
        group_rows,
        fieldnames_for_group_rows(include_upstream, include_paper),
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "config": str(args.config),
        "repo_root": str(args.repo_root),
        "output_root": str(args.output_root),
        "input_size": args.input_size,
        "fp32": bool(args.fp32),
        "models": summary_rows,
        "manifest_metadata": manifest["metadata"],
    }
    save_json(scope_root / "summary.json", payload)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_runtime_cache(args.output_root)

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg["__config_path__"] = str(args.config)

    model_names = parse_model_names(args.models)
    scopes = parse_scopes(args.stream_scope)
    include_upstream = args.metric_suite in {"upstream", "both"}
    include_paper = args.metric_suite in {"paper", "both"}
    device = resolve_device(args.device)

    manifest = export_degraded_dataset(
        cfg,
        args.output_root / "dataset_export",
        overwrite_export=bool(args.overwrite_export),
        debug_max_trajectories=args.debug_max_trajectories,
    )

    for mode in scopes:
        summary_rows: list[dict[str, object]] = []
        group_rows: list[dict[str, object]] = []
        for model_name in model_names:
            checkpoint_path = DEFAULT_CHECKPOINTS[model_name]
            summary_row, model_group_rows = run_scope_benchmark(
                args=args,
                manifest=manifest,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                mode=mode,
                device=device,
                include_upstream=include_upstream,
                include_paper=include_paper,
            )
            summary_rows.append(summary_row)
            group_rows.extend(model_group_rows)
        save_scope_reports(
            args.output_root,
            mode,
            summary_rows,
            group_rows,
            include_upstream=include_upstream,
            include_paper=include_paper,
            manifest=manifest,
            args=args,
        )

    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "dataset_export": str(args.output_root / "dataset_export"),
                "summary_root": str(args.output_root / "summary"),
                "models": model_names,
                "scopes": scopes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
