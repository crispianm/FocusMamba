"""RoboDepth evaluation helpers for VDA-style models."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics.depth_metrics import compute_depth_metrics


KITTI_C_CORRUPTIONS: tuple[str, ...] = (
    "brightness",
    "dark",
    "fog",
    "frost",
    "snow",
    "contrast",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "elastic_transform",
    "color_quant",
    "gaussian_noise",
    "impulse_noise",
    "shot_noise",
    "iso_noise",
    "pixelate",
    "jpeg_compression",
)

KITTI_C_FAMILIES: dict[str, tuple[str, ...]] = {
    "weather_lighting": ("brightness", "dark", "fog", "frost", "snow", "contrast"),
    "blur_geometry": (
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "elastic_transform",
    ),
    "noise_processing": (
        "color_quant",
        "gaussian_noise",
        "impulse_noise",
        "shot_noise",
        "iso_noise",
        "pixelate",
        "jpeg_compression",
    ),
}

MONODEPTH2_MONO_BASELINE: dict[str, dict[str, float]] = {
    "DEE1": {
        "clean": 0.238,
        "brightness": 0.259,
        "dark": 0.561,
        "fog": 0.311,
        "frost": 0.553,
        "snow": 1.023,
        "contrast": 0.373,
        "defocus_blur": 0.487,
        "glass_blur": 0.484,
        "motion_blur": 0.433,
        "zoom_blur": 0.402,
        "elastic_transform": 0.258,
        "color_quant": 0.386,
        "gaussian_noise": 0.768,
        "impulse_noise": 0.779,
        "shot_noise": 0.681,
        "iso_noise": 0.776,
        "pixelate": 0.289,
        "jpeg_compression": 0.391,
    },
    "DEE2": {
        "clean": 0.119,
        "brightness": 0.130,
        "dark": 0.280,
        "fog": 0.155,
        "frost": 0.277,
        "snow": 0.511,
        "contrast": 0.187,
        "defocus_blur": 0.244,
        "glass_blur": 0.242,
        "motion_blur": 0.216,
        "zoom_blur": 0.201,
        "elastic_transform": 0.129,
        "color_quant": 0.193,
        "gaussian_noise": 0.384,
        "impulse_noise": 0.389,
        "shot_noise": 0.340,
        "iso_noise": 0.388,
        "pixelate": 0.145,
        "jpeg_compression": 0.196,
    },
    "DEE3": {
        "clean": 0.131,
        "brightness": 0.140,
        "dark": 0.330,
        "fog": 0.166,
        "frost": 0.327,
        "snow": 1.058,
        "contrast": 0.199,
        "defocus_blur": 0.271,
        "glass_blur": 0.270,
        "motion_blur": 0.236,
        "zoom_blur": 0.221,
        "elastic_transform": 0.141,
        "color_quant": 0.209,
        "gaussian_noise": 0.551,
        "impulse_noise": 0.563,
        "shot_noise": 0.442,
        "iso_noise": 0.557,
        "pixelate": 0.158,
        "jpeg_compression": 0.215,
    },
}


class RoboDepthKITTIDataset(Dataset):
    """Minimal KITTI-C style dataset loader for RoboDepth evaluation."""

    def __init__(
        self,
        *,
        root: str | Path,
        split: str = "eigen",
        corruption: str | None = None,
        severity: int | None = None,
        max_samples: int | None = None,
        clean_subdir: str = "kitti_data",
        corrupted_subdir: str = "kitti_c",
        split_file: str | Path | None = None,
        gt_path: str | Path | None = None,
        use_eigen_crop: bool = True,
    ) -> None:
        self.root = Path(root)
        self.clean_subdir = str(clean_subdir)
        self.corrupted_subdir = str(corrupted_subdir)
        self.corruption = corruption
        self.severity = severity
        self.use_eigen_crop = bool(use_eigen_crop)

        if split_file is None:
            split_file = self.root / "splits" / split / "test_files.txt"
        else:
            split_file = Path(split_file)
        if gt_path is None:
            gt_path = self.root / "splits" / split / "gt_depths.npz"
        else:
            gt_path = Path(gt_path)

        if not split_file.is_file():
            raise FileNotFoundError(f"RoboDepth split file not found: {split_file}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"RoboDepth GT depth file not found: {gt_path}")

        self.samples = [
            line.strip() for line in split_file.read_text().splitlines() if line.strip()
        ]
        self.gt_depths = np.load(gt_path, allow_pickle=True)["data"]
        if len(self.samples) != len(self.gt_depths):
            raise ValueError(
                f"Split/GT length mismatch for RoboDepth KITTI-C: {len(self.samples)} files vs {len(self.gt_depths)} depths"
            )
        if max_samples is not None:
            limit = max(int(max_samples), 0)
            self.samples = self.samples[:limit]
            self.gt_depths = self.gt_depths[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _parse_sample(line: str) -> tuple[str, int, str]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Unsupported RoboDepth KITTI split line: {line!r}")
        return parts[0], int(parts[1]), parts[2]

    def _image_path(self, folder: str, frame_idx: int, side: str) -> Path:
        side_folder = "image_02" if side.lower().startswith("l") else "image_03"
        frame_name = f"{frame_idx:010d}.png"
        if self.corruption is None:
            return (
                self.root
                / self.clean_subdir
                / folder
                / side_folder
                / "data"
                / frame_name
            )
        return (
            self.root
            / self.corrupted_subdir
            / self.corruption
            / str(int(self.severity or 1))
            / self.clean_subdir
            / folder
            / side_folder
            / "data"
            / frame_name
        )

    @staticmethod
    def _crop_mask(height: int, width: int, *, use_eigen_crop: bool) -> np.ndarray:
        mask = np.ones((height, width), dtype=bool)
        if use_eigen_crop and height > 16 and width > 16:
            crop = np.array(
                [
                    0.40810811 * height,
                    0.99189189 * height,
                    0.03594771 * width,
                    0.96405229 * width,
                ]
            ).astype(np.int32)
            mask[:] = False
            mask[crop[0] : crop[1], crop[2] : crop[3]] = True
        return mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        folder, frame_idx, side = self._parse_sample(self.samples[idx])
        image_path = self._image_path(folder, frame_idx, side)
        if not image_path.is_file():
            raise FileNotFoundError(f"RoboDepth image not found: {image_path}")

        image = (
            np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        )
        gt_depth = np.asarray(self.gt_depths[idx], dtype=np.float32)
        crop_mask = self._crop_mask(
            gt_depth.shape[0], gt_depth.shape[1], use_eigen_crop=self.use_eigen_crop
        )
        valid_mask = (
            crop_mask & np.isfinite(gt_depth) & (gt_depth > 1e-3) & (gt_depth < 80.0)
        )

        image_t = torch.from_numpy(image).permute(2, 0, 1).float()
        depth_t = torch.from_numpy(gt_depth).unsqueeze(0).float()
        mask_t = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0)
        return {
            "image": image_t,
            "depth": depth_t,
            "mask": mask_t,
        }


def _resolve_robodepth_cfg(config: dict[str, Any]) -> dict[str, Any]:
    validation_cfg = config.get("validation", {}) or {}
    robodepth_cfg = validation_cfg.get("robodepth", config.get("robodepth", {})) or {}
    return deepcopy(robodepth_cfg)


def _prepare_video_batch(
    images: torch.Tensor,
    *,
    repeat_frames: int,
    inference_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    original_hw = (int(images.shape[-2]), int(images.shape[-1]))
    model_images = images
    if inference_size is not None and tuple(inference_size) != original_hw:
        model_images = F.interpolate(
            images, size=inference_size, mode="bilinear", align_corners=False
        )
    frames = model_images.unsqueeze(2).repeat(1, 1, max(int(repeat_frames), 1), 1, 1)
    return frames, original_hw


@torch.no_grad()
def _predict_depth_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    repeat_frames: int,
    inference_size: tuple[int, int] | None,
    model_mode: str,
    use_amp: bool,
) -> torch.Tensor:
    frames, original_hw = _prepare_video_batch(
        images.to(device), repeat_frames=repeat_frames, inference_size=inference_size
    )
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(frames, mode=model_mode)
    else:
        outputs = model(frames, mode=model_mode)
    depth = outputs["depth"][:, :, repeat_frames // 2].float()
    if tuple(depth.shape[-2:]) != original_hw:
        depth = F.interpolate(
            depth, size=original_hw, mode="bilinear", align_corners=False
        )
    return depth


def _dee_metrics(metrics: dict[str, float]) -> dict[str, float]:
    abs_rel = float(metrics["abs_rel"])
    delta1 = float(metrics["delta1"])
    dee1 = abs_rel - delta1 + 1.0
    dee2 = 0.5 * dee1
    dee3 = abs_rel / max(delta1, 1e-6)
    return {
        "dee1": dee1,
        "dee2": dee2,
        "dee3": dee3,
    }


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return float("nan")
    return float(sum(seq) / len(seq))


def _safe_ratio(numer: float, denom: float) -> float:
    denom = float(denom)
    if abs(denom) < 1e-8:
        return float("nan")
    return float(numer / denom)


def _load_baseline(name_or_path: str | None) -> dict[str, dict[str, float]] | None:
    if name_or_path is None:
        return None
    token = str(name_or_path).strip().lower()
    if token in {"", "none", "off"}:
        return None
    if token == "monodepth2_mono":
        return deepcopy(MONODEPTH2_MONO_BASELINE)
    path = Path(name_or_path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    raise FileNotFoundError(f"Unknown RoboDepth baseline reference: {name_or_path}")


@torch.no_grad()
def _evaluate_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    repeat_frames: int,
    inference_size: tuple[int, int] | None,
    model_mode: str,
    use_amp: bool,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=max(int(num_workers), 0),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    sums = {
        "abs_rel": 0.0,
        "sq_rel": 0.0,
        "rmse": 0.0,
        "rmse_log": 0.0,
        "si_log": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0,
    }
    batches = 0
    for batch in loader:
        pred = _predict_depth_batch(
            model,
            batch["image"],
            device=device,
            repeat_frames=repeat_frames,
            inference_size=inference_size,
            model_mode=model_mode,
            use_amp=use_amp,
        )
        metrics = compute_depth_metrics(
            pred.unsqueeze(2),
            batch["depth"].to(device).unsqueeze(2),
            mask=batch["mask"].to(device).unsqueeze(2),
        )
        for key in sums:
            sums[key] += float(metrics[key])
        batches += 1

    if batches == 0:
        return {key: float("nan") for key in sums}
    averaged = {key: value / batches for key, value in sums.items()}
    averaged.update(_dee_metrics(averaged))
    averaged["num_batches"] = float(batches)
    averaged["num_samples"] = float(len(dataset))
    return averaged


@torch.no_grad()
def _measure_latency(
    model: torch.nn.Module,
    *,
    device: torch.device,
    repeat_frames: int,
    inference_size: tuple[int, int],
    model_mode: str,
    use_amp: bool,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    dummy = torch.randn(
        1,
        3,
        max(int(repeat_frames), 1),
        inference_size[0],
        inference_size[1],
        device=device,
    )
    for _ in range(max(int(warmup), 0)):
        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                _ = model(dummy, mode=model_mode)
        else:
            _ = model(dummy, mode=model_mode)
    if device.type == "cuda":
        torch.cuda.synchronize()

    import time

    start = time.perf_counter()
    for _ in range(max(int(iterations), 1)):
        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                _ = model(dummy, mode=model_mode)
        else:
            _ = model(dummy, mode=model_mode)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_frames = max(int(iterations), 1) * max(int(repeat_frames), 1)
    fps = total_frames / max(elapsed, 1e-6)
    return {
        "fps": float(fps),
        "ms_per_frame": float(elapsed / total_frames * 1000.0),
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "robodepth_summary.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    rows: list[dict[str, Any]] = []
    clean_metrics = report["details"]["clean"]
    rows.append({"corruption": "clean", "severity": 0, **clean_metrics})
    for corruption, severity_metrics in report["details"]["per_corruption"].items():
        for severity, metrics in severity_metrics.items():
            rows.append({"corruption": corruption, "severity": severity, **metrics})

    if rows:
        tsv_path = output_dir / "robodepth_per_corruption.tsv"
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with tsv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    family_rows = [
        {"family": family_name, **metrics}
        for family_name, metrics in report["details"].get("family_means", {}).items()
    ]
    if family_rows:
        family_tsv_path = output_dir / "robodepth_family_means.tsv"
        fieldnames = sorted({key for row in family_rows for key in row.keys()})
        with family_tsv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in family_rows:
                writer.writerow(row)


def run_robodepth_eval(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run RoboDepth-style KITTI-C evaluation for the current model."""

    robodepth_cfg = _resolve_robodepth_cfg(config)
    if not robodepth_cfg.get("enabled", False):
        raise NotImplementedError("validation.robodepth.enabled is false")

    dataset_kind = str(robodepth_cfg.get("dataset", "kitti_c")).strip().lower()
    if dataset_kind != "kitti_c":
        raise NotImplementedError(
            f"Unsupported RoboDepth dataset kind: {dataset_kind!r}"
        )

    root = Path(robodepth_cfg.get("root", "")).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"RoboDepth root does not exist: {root}")

    split = str(robodepth_cfg.get("split", "eigen"))
    repeat_frames = int(
        robodepth_cfg.get(
            "repeat_frames", config.get("data", {}).get("val_num_frames", 8)
        )
    )
    inference_hw_raw = robodepth_cfg.get("inference_size")
    inference_size = tuple(inference_hw_raw) if inference_hw_raw is not None else None
    if inference_size is None:
        image_size = config.get("data", {}).get("image_size", [192, 640])
        inference_size = (int(image_size[0]), int(image_size[1]))
    batch_size = int(robodepth_cfg.get("batch_size", 1))
    num_workers = int(robodepth_cfg.get("num_workers", 0))
    max_samples = robodepth_cfg.get("max_samples", None)
    model_mode = str(
        robodepth_cfg.get("model_mode", config.get("model", {}).get("mode", "offline"))
    )
    use_amp = bool(robodepth_cfg.get("use_amp", device.type == "cuda"))
    corruptions = tuple(robodepth_cfg.get("corruptions", KITTI_C_CORRUPTIONS))
    severities = tuple(int(s) for s in robodepth_cfg.get("severities", [1, 2, 3, 4, 5]))
    baseline = _load_baseline(robodepth_cfg.get("baseline_name", "monodepth2_mono"))

    clean_dataset = RoboDepthKITTIDataset(
        root=root,
        split=split,
        corruption=None,
        severity=None,
        max_samples=max_samples,
        clean_subdir=str(robodepth_cfg.get("clean_subdir", "kitti_data")),
        corrupted_subdir=str(robodepth_cfg.get("corrupted_subdir", "kitti_c")),
        split_file=robodepth_cfg.get("split_file"),
        gt_path=robodepth_cfg.get("gt_path"),
        use_eigen_crop=bool(robodepth_cfg.get("use_eigen_crop", True)),
    )
    clean_metrics = _evaluate_dataset(
        model,
        clean_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        repeat_frames=repeat_frames,
        inference_size=inference_size,
        model_mode=model_mode,
        use_amp=use_amp,
    )

    per_corruption: dict[str, dict[int, dict[str, float]]] = {}
    corruption_means: dict[str, dict[str, float]] = {}
    for corruption in corruptions:
        severity_results: dict[int, dict[str, float]] = {}
        for severity in severities:
            dataset = RoboDepthKITTIDataset(
                root=root,
                split=split,
                corruption=corruption,
                severity=severity,
                max_samples=max_samples,
                clean_subdir=str(robodepth_cfg.get("clean_subdir", "kitti_data")),
                corrupted_subdir=str(robodepth_cfg.get("corrupted_subdir", "kitti_c")),
                split_file=robodepth_cfg.get("split_file"),
                gt_path=robodepth_cfg.get("gt_path"),
                use_eigen_crop=bool(robodepth_cfg.get("use_eigen_crop", True)),
            )
            severity_results[severity] = _evaluate_dataset(
                model,
                dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                repeat_frames=repeat_frames,
                inference_size=inference_size,
                model_mode=model_mode,
                use_amp=use_amp,
            )
        per_corruption[corruption] = severity_results
        corruption_means[corruption] = {
            key: _mean(result[key] for result in severity_results.values())
            for key in next(iter(severity_results.values())).keys()
        }

    family_means: dict[str, dict[str, float]] = {}
    for family_name, family_corruptions in KITTI_C_FAMILIES.items():
        family_entries = [
            corruption_means[name]
            for name in family_corruptions
            if name in corruption_means
        ]
        if not family_entries:
            continue
        family_means[family_name] = {
            key: _mean(entry[key] for entry in family_entries)
            for key in family_entries[0].keys()
        }

    summary = {
        "robodepth_clean_dee1": clean_metrics["dee1"],
        "robodepth_clean_dee2": clean_metrics["dee2"],
        "robodepth_clean_dee3": clean_metrics["dee3"],
        "robodepth_clean_abs_rel": clean_metrics["abs_rel"],
        "robodepth_clean_delta1": clean_metrics["delta1"],
        "robodepth_num_samples": clean_metrics["num_samples"],
        "robodepth_num_corruptions": float(len(corruption_means)),
    }

    summary["robodepth_corruption_dee1_mean"] = _mean(
        metrics["dee1"] for metrics in corruption_means.values()
    )
    summary["robodepth_corruption_dee2_mean"] = _mean(
        metrics["dee2"] for metrics in corruption_means.values()
    )
    summary["robodepth_corruption_dee3_mean"] = _mean(
        metrics["dee3"] for metrics in corruption_means.values()
    )
    summary["robodepth_dee1"] = summary["robodepth_corruption_dee1_mean"]
    summary["robodepth_dee2"] = summary["robodepth_corruption_dee2_mean"]
    summary["robodepth_dee3"] = summary["robodepth_corruption_dee3_mean"]
    summary["robodepth_mrr1"] = _mean(
        _safe_ratio(1.0 - metrics["dee1"], 1.0 - clean_metrics["dee1"])
        for metrics in corruption_means.values()
    )
    summary["robodepth_mrr2"] = _mean(
        _safe_ratio(1.0 - metrics["dee2"], 1.0 - clean_metrics["dee2"])
        for metrics in corruption_means.values()
    )
    summary["robodepth_mrr3"] = _mean(
        _safe_ratio(1.0 - metrics["dee3"], 1.0 - clean_metrics["dee3"])
        for metrics in corruption_means.values()
    )

    if baseline is not None:
        summary["robodepth_mce1"] = _mean(
            _safe_ratio(corruption_means[name]["dee1"], baseline["DEE1"][name])
            for name in corruption_means
            if name in baseline["DEE1"]
        )
        summary["robodepth_mce2"] = _mean(
            _safe_ratio(corruption_means[name]["dee2"], baseline["DEE2"][name])
            for name in corruption_means
            if name in baseline["DEE2"]
        )
        summary["robodepth_mce3"] = _mean(
            _safe_ratio(corruption_means[name]["dee3"], baseline["DEE3"][name])
            for name in corruption_means
            if name in baseline["DEE3"]
        )
        summary["robodepth_rmce2"] = _mean(
            _safe_ratio(
                clean_metrics["dee2"] - corruption_means[name]["dee2"],
                baseline["DEE2"]["clean"] - baseline["DEE2"][name],
            )
            for name in corruption_means
            if name in baseline["DEE2"]
        )

    latency = _measure_latency(
        model,
        device=device,
        repeat_frames=repeat_frames,
        inference_size=inference_size,
        model_mode=model_mode,
        use_amp=use_amp,
        warmup=int(robodepth_cfg.get("latency_warmup", 5)),
        iterations=int(robodepth_cfg.get("latency_iterations", 20)),
    )
    summary["robodepth_fps"] = latency["fps"]
    summary["robodepth_ms_per_frame"] = latency["ms_per_frame"]
    target_fps = float(
        robodepth_cfg.get(
            "target_fps", config.get("inference", {}).get("target_fps", 0.0)
        )
        or 0.0
    )
    if target_fps > 0.0:
        summary["robodepth_target_fps"] = target_fps
        summary["robodepth_fps_margin"] = summary["robodepth_fps"] - target_fps
        summary["robodepth_meets_target_fps"] = (
            1.0 if summary["robodepth_fps"] >= target_fps else 0.0
        )

    family_scalar_metrics = {}
    for family_name, metrics in family_means.items():
        family_key = family_name.lower()
        for metric_name in ("abs_rel", "delta1", "dee1", "dee2", "dee3"):
            if metric_name in metrics:
                family_scalar_metrics[
                    f"robodepth_family_{family_key}_{metric_name}"
                ] = float(metrics[metric_name])
    summary.update(family_scalar_metrics)

    report = {
        **summary,
        "details": {
            "clean": clean_metrics,
            "per_corruption": per_corruption,
            "corruption_means": corruption_means,
            "family_means": family_means,
        },
    }
    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    if resolved_output_dir is not None:
        _write_report(resolved_output_dir, report)
        report["robodepth_report_dir"] = str(resolved_output_dir)
    return report
