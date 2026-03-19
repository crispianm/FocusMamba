"""Per-degradation proxy evaluation on degraded TartanAir validation clips."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from dataloader.degradation import build_degradation
from dataloader.tartanair_v2 import TartanAirV2Dataset
from evaluation.metrics.depth_metrics import compute_depth_metrics


DEFAULT_FAMILIES = (
    "lowlight_noise",
    "blur",
    "processing_artifacts",
    "sensor_drift_events",
)


@torch.no_grad()
def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    model_mode: str,
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    batches = 0
    for batch in loader:
        frames = batch["degraded_frames"].to(device)
        gt_depth = (
            batch["depth"].to(device).unsqueeze(2)
            if batch["depth"].ndim == 4
            else batch["depth"].to(device)
        )
        mask = (
            batch["mask"].to(device).unsqueeze(2)
            if batch["mask"].ndim == 4
            else batch["mask"].to(device)
        )
        outputs = model(frames, mode=model_mode)
        pred = outputs["depth"]
        metrics = compute_depth_metrics(pred, gt_depth, mask=mask)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        batches += 1

    if batches == 0:
        return {}
    return {key: value / batches for key, value in sums.items()}


def run_degradation_sweep(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate VDA on a few local degradation families as a fast proxy."""

    data_cfg = config.get("data", {}) or {}
    if str(data_cfg.get("dataset", "tartanair_v2")).lower() != "tartanair_v2":
        raise NotImplementedError(
            "Per-degradation sweep is currently implemented for TartanAirV2 only"
        )

    eval_cfg = (config.get("evaluation", {}) or {}).get("per_degradation", {}) or {}
    families = tuple(eval_cfg.get("families", DEFAULT_FAMILIES))
    severities = tuple(float(s) for s in eval_cfg.get("severities", [0.25, 0.5, 1.0]))
    batch_size = int(eval_cfg.get("batch_size", 1))
    max_trajectories = eval_cfg.get(
        "max_val_trajectories", data_cfg.get("max_val_trajectories", 1)
    )
    model_mode = str(
        eval_cfg.get("model_mode", config.get("model", {}).get("mode", "offline"))
    )

    results: Dict[str, Any] = {}
    for family in families:
        family_results: Dict[str, Any] = {}
        for severity in severities:
            eval_config = deepcopy(config)
            eval_deg_cfg = eval_config.setdefault("degradation", {})
            eval_deg_cfg["enabled"] = True
            eval_deg_cfg["policy"] = str(eval_deg_cfg.get("policy", "v2_lowlight"))
            eval_deg_cfg["eval_preset"] = family
            eval_deg_cfg["eval_severity"] = severity
            eval_deg_cfg["summary"] = False
            eval_deg_cfg["return_params"] = False
            degradation = build_degradation(eval_config)

            dataset = TartanAirV2Dataset(
                root=str(data_cfg["root"]),
                num_frames=int(
                    data_cfg.get("val_num_frames", data_cfg.get("num_frames", 8))
                ),
                image_size=tuple(data_cfg.get("image_size", (256, 256))),
                max_trajectories=max_trajectories,
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
                return_degradation_metadata=False,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            family_results[f"{severity:.2f}"] = _evaluate_loader(
                model,
                loader,
                device=device,
                model_mode=model_mode,
            )
        results[family] = family_results
    return results
