#!/usr/bin/env python3
"""Generate focused-redirection experiment suites."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import yaml


def deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


TEACHERS = [
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


def _distill_cfg(strategy: str) -> dict:
    return {
        "enabled": True,
        "target_mode": "metric",
        "strategy": strategy,
        "teachers": [{"name": t["name"], "weight": 1.0} for t in TEACHERS],
        "confidence_weighted": True,
        "lambda_si": 0.5,
        "calibration": {"enabled": True, "min_depth": 0.1, "min_valid_pixels": 64, "use_gt": True},
        "aggregate": {"weight": 0.5, "beta": 6.0},
        "auxiliary": {
            "vda_weight": 0.3,
            "da3_weight": 0.1,
            "depth_pro_weight": 0.1,
            "vda_temporal_weight": 0.5,
            "structure_weight": 0.5,
            "gradient_scales": 1,
        },
    }


BASE = {
    "model": {
        "type": "video_depth_anything",
        "variant": "small",
        "num_frames": 16,
        "positional_encoding": "ape",
        "checkpoint_path": "checkpoints/metric_video_depth_anything_vits.pth",
        "strict_checkpoint": False,
        "mode": "streaming_emulated",
        "stream_max_cache_len": 16,
        "stream_reset_interval": 0,
        "state_gate_enabled": False,
        "state_gate_reduction": 8,
        "prefilter_enabled": False,
        "prefilter_type": "fast_classical",
        "prefilter_target_mean": [0.485, 0.456, 0.406],
        "prefilter_target_std": [0.229, 0.224, 0.225],
        "prefilter_kernel_size": 5,
        "prefilter_sigma": 1.0,
        "prefilter_denoise_init": 0.20,
        "prefilter_sharpen_init": 0.10,
        "prefilter_learnable": False,
    },
    "data": {
        "dataset": "tartanair_v2",
        "root": "/projects/b5dh/data/tartanair-v2",
        "train_num_frames": 8,
        "val_num_frames": 16,
        "image_size": [392, 392],
        "max_trajectories": None,
        "max_train_trajectories": 64,
        "max_val_trajectories": 8,
        "clip_stride": 8,
        "frame_stride": 1,
        "val_fraction": 0.1,
        "seed": 42,
        "difficulty": "Data_easy",
        "camera": "lcam_front",
        "max_depth": 80.0,
        "envs": None,
        "teacher_cache_dir": "/projects/b5dh/data/teacher_cache/tartanair_v2",
    },
    "teachers": TEACHERS,
    "distillation": _distill_cfg("heterogeneous_mtkd"),
    "degradation": {
        "enabled": True,
        "lux_level": 10.0,
        "poisson_scale": 0.02,
        "gaussian_std": 0.01,
        "blur_kernel_size": 15,
        "blur_intensity": 1.0,
        "temporal_variance": 0.05,
        "apply_brightness_drop": True,
        "curriculum": False,
    },
    "training": {
        "max_epochs": 3,
        "batch_size": 1,
        "learning_rate": 5.0e-05,
        "backbone_lr_multiplier": 0.3,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "grad_clip": 1.0,
        "precision": "bf16",
        "val_every_n_epochs": 1,
        "selection_metric": "abs_rel",
        "ema_decay": 0.999,
        "num_workers": 8,
        "val_num_workers": 0,
        "gradient_accumulation_steps": 4,
        "compile": False,
        "ddp_find_unused_parameters": True,
        "skip_live_teachers": True,
        "cpu_prefetch": True,
        "cpu_prefetch_batches": 1,
        "val_cpu_prefetch": False,
        "log_images_every_n_steps": 1000,
        "log_images_max_batch": 2,
        "step_log_every_n_steps": 100,
    },
    "validation": {"use_teacher_signals": False},
    "loss": {
        "training_target": "metric",
        "ssi_weight": 0.25,
        "temporal_weight": 1.0,
        "temporal_threshold": 0.05,
        "temporal_log_space": False,
        "ssi_mode": "l2",
        "distillation_weight": 0.15,
        "metric_l1_weight": 0.5,
        "metric_log_l1_weight": 1.0,
    },
}


def _recipe(strategy: str) -> dict:
    return {"distillation": _distill_cfg(strategy)}


def _stats_align(target_mean: list[float] | None = None, target_std: list[float] | None = None) -> dict:
    model = {
        "prefilter_enabled": True,
        "prefilter_type": "stats_align",
        "prefilter_learnable": False,
    }
    if target_mean is not None:
        model["prefilter_target_mean"] = target_mean
    if target_std is not None:
        model["prefilter_target_std"] = target_std
    return {"model": model}


VARIANTS = [
    {"name": "00_anchor_baseline", "job": "fABs", "overrides": _recipe("anchor_vda")},
    {"name": "01_anchor_stats_align", "job": "fASt", "overrides": deep_update(_recipe("anchor_vda"), _stats_align())},
    {"name": "02_anchor_gate_only", "job": "fAGt", "overrides": deep_update(_recipe("anchor_vda"), {"model": {"state_gate_enabled": True}})},
    {"name": "03_anchor_stats_align_gate", "job": "fASG", "overrides": deep_update(deepcopy(_recipe("anchor_vda")), deep_update(_stats_align(), {"model": {"state_gate_enabled": True}}))},
    {"name": "04_anchor_reset8", "job": "fAR8", "overrides": deep_update(_recipe("anchor_vda"), {"model": {"stream_reset_interval": 8}})},
    {"name": "05_anchor_reset16", "job": "fAR6", "overrides": deep_update(_recipe("anchor_vda"), {"model": {"stream_reset_interval": 16}})},
    {"name": "06_anchor_shortmem8", "job": "fAK8", "overrides": deep_update(_recipe("anchor_vda"), {"model": {"stream_max_cache_len": 8}})},
    {"name": "07_hetero_baseline", "job": "fHBs", "overrides": _recipe("heterogeneous_mtkd")},
    {"name": "08_hetero_stats_align", "job": "fHSt", "overrides": deep_update(_recipe("heterogeneous_mtkd"), _stats_align())},
    {"name": "09_hetero_gate_only", "job": "fHGt", "overrides": deep_update(_recipe("heterogeneous_mtkd"), {"model": {"state_gate_enabled": True}})},
    {"name": "10_hetero_stats_align_gate", "job": "fHSG", "overrides": deep_update(deepcopy(_recipe("heterogeneous_mtkd")), deep_update(_stats_align(), {"model": {"state_gate_enabled": True}}))},
    {"name": "11_hetero_reset8", "job": "fHR8", "overrides": deep_update(_recipe("heterogeneous_mtkd"), {"model": {"stream_reset_interval": 8}})},
    {"name": "12_hetero_reset16", "job": "fHR6", "overrides": deep_update(_recipe("heterogeneous_mtkd"), {"model": {"stream_reset_interval": 16}})},
    {"name": "13_hetero_shortmem8", "job": "fHK8", "overrides": deep_update(_recipe("heterogeneous_mtkd"), {"model": {"stream_max_cache_len": 8}})},
    {"name": "14_hetero_stats_align_gate_laststage", "job": "fHGL", "overrides": deep_update(deepcopy(_recipe("heterogeneous_mtkd")), deep_update(_stats_align(), {"model": {"state_gate_enabled": True, "state_gate_stage_mask": [False, False, False, True]}}))},
    {"name": "15_hetero_stats_align_gate_temporalhead", "job": "fHGH", "overrides": deep_update(deepcopy(_recipe("heterogeneous_mtkd")), deep_update(_stats_align(), {"model": {"state_gate_enabled": True}, "training": {"freeze_backbone": True, "unfreeze_prefixes": ["model.head.motion_modules", "model.head.stage_gates"]}}))},
    {"name": "16_hetero_stats_align_curriculum_moderate", "job": "fHCM", "overrides": deep_update(_recipe("heterogeneous_mtkd"), deep_update(_stats_align(), {"degradation": {"curriculum": {"warmup_epochs": 1, "max_severity_epoch": 2, "schedule": "linear", "warmup_start_scale": 0.5, "min_scale": 0.5, "max_scale": 1.0}}}))},
    {"name": "17_hetero_stats_align_gate_curriculum_mixed", "job": "fHGM", "overrides": deep_update(deepcopy(_recipe("heterogeneous_mtkd")), deep_update(_stats_align(), {"model": {"state_gate_enabled": True}, "degradation": {"curriculum": {"warmup_epochs": 0, "max_severity_epoch": 2, "schedule": "linear", "warmup_start_scale": 1.0, "min_scale": 1.0, "max_scale": 1.0, "mix_clean_probability_start": 0.50, "mix_clean_probability_end": 0.20}}}))},
    {"name": "18_hetero_stats_align_temporal_smooth", "job": "fHTS", "overrides": deep_update(_recipe("heterogeneous_mtkd"), deep_update(_stats_align(), {"loss": {"temporal_weight": 2.5, "temporal_threshold": 0.03}}))},
    {"name": "19_hetero_stats_align_clean_consistency", "job": "fHCC", "overrides": deep_update(_recipe("heterogeneous_mtkd"), deep_update(_stats_align(), {"loss": {"auxiliary": {"clean_depth_consistency_weight": 0.05, "clean_depth_consistency_mode": "log_l1"}}}))},
    {"name": "20_hetero_stats_align_neutral_stats", "job": "fHSN", "overrides": deep_update(_recipe("heterogeneous_mtkd"), _stats_align([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]))},
    {"name": "21_hetero_stats_align_bright_stats", "job": "fHSB", "overrides": deep_update(_recipe("heterogeneous_mtkd"), _stats_align([0.58, 0.56, 0.54], [0.24, 0.23, 0.22]))},
]


def _front_adapter() -> dict:
    return {
        "model": {
            "prefilter_enabled": True,
            "prefilter_type": "stats_guided_front_adapter",
            "front_adapter_hidden": 16,
            "front_adapter_blocks": 2,
            "front_adapter_use_stats_align": True,
            "front_adapter_use_se": True,
            "prefilter_learnable": False,
        }
    }


def _pre_temporal_adapter() -> dict:
    return {
        "model": {
            "pre_temporal_stage_adapter_enabled": True,
            "pre_temporal_stage_adapter_stages": ["layer3", "layer4"],
            "pre_temporal_stage_adapter_bottleneck_ratio": 4,
        }
    }


def _clean_consistency() -> dict:
    return {
        "loss": {
            "auxiliary": {
                "clean_depth_consistency_weight": 0.05,
                "clean_depth_consistency_mode": "log_l1",
            }
        }
    }


def _mixed_curriculum() -> dict:
    return {
        "degradation": {
            "curriculum": {
                "warmup_epochs": 0,
                "max_severity_epoch": 2,
                "schedule": "linear",
                "warmup_start_scale": 1.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
                "mix_clean_probability_start": 0.50,
                "mix_clean_probability_end": 0.20,
            }
        }
    }


FRONT_ADAPTER_STAGEB_VARIANTS = [
    {"name": "00_anchor_baseline", "job": "fa00", "overrides": _recipe("anchor_vda")},
    {"name": "01_anchor_stats_align", "job": "fa01", "overrides": deep_update(_recipe("anchor_vda"), _stats_align())},
    {"name": "02_anchor_front_adapter", "job": "fa02", "overrides": deep_update(_recipe("anchor_vda"), _front_adapter())},
    {
        "name": "03_anchor_front_adapter_gate",
        "job": "fa03",
        "overrides": deep_update(deepcopy(_recipe("anchor_vda")), deep_update(_front_adapter(), {"model": {"state_gate_enabled": True}})),
    },
    {
        "name": "04_anchor_pre_temporal_adapter",
        "job": "fa04",
        "overrides": deep_update(_recipe("anchor_vda"), _pre_temporal_adapter()),
    },
    {
        "name": "05_anchor_pre_temporal_adapter_stats_align",
        "job": "fa05",
        "overrides": deep_update(deepcopy(_recipe("anchor_vda")), deep_update(_pre_temporal_adapter(), _stats_align())),
    },
    {
        "name": "06_anchor_front_adapter_clean_consistency",
        "job": "fa06",
        "overrides": deep_update(deepcopy(_recipe("anchor_vda")), deep_update(_front_adapter(), _clean_consistency())),
    },
    {
        "name": "07_anchor_front_adapter_curriculum_mixed",
        "job": "fa07",
        "overrides": deep_update(deepcopy(_recipe("anchor_vda")), deep_update(_front_adapter(), _mixed_curriculum())),
    },
]


SUITES = {
    "paperlock": {
        "default_suite_name": "focused_redirection_paperlock_20260310",
        "variants": VARIANTS,
        "notes": [
            "Paper-lock focused-redirection sweep.",
            "Covers exact main-table rows plus only hypothesis-relevant refinement variants.",
            "Teacher diversity is treated as a recipe choice, not the main novelty.",
        ],
    },
    "front_adapter_stageb": {
        "default_suite_name": "focused_front_adapter_stageb_20260312",
        "variants": FRONT_ADAPTER_STAGEB_VARIANTS,
        "notes": [
            "Stage-B architecture screen for lightweight front and pre-temporal adapters.",
            "Uses anchor_vda as the fixed baseline recipe to isolate architecture effects.",
            "This suite is for matched subset screening before full-data promotion.",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--suite-kind", choices=tuple(SUITES), default="paperlock")
    parser.add_argument("--profile", choices=("screen", "promote"), default="screen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    suite_spec = SUITES[args.suite_kind]
    suite_name = str(args.suite_name or suite_spec["default_suite_name"])
    cfg_dir = project_root / "configs" / "experiments" / "generated" / suite_name
    run_root = project_root / "runs" / suite_name
    ckpt_root = project_root / "checkpoints" / suite_name
    jobs_tsv = cfg_dir / "jobs.tsv"
    manifest = cfg_dir / "manifest.yaml"

    cfg_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    jobs: list[tuple[str, str]] = []

    variants = suite_spec["variants"]
    for variant in variants:
        cfg = deepcopy(BASE)
        deep_update(cfg, deepcopy(variant["overrides"]))
        if args.suite_kind == "front_adapter_stageb":
            if args.profile != "screen":
                raise ValueError("front_adapter_stageb currently only supports --profile screen")
            cfg["training"]["max_epochs"] = 6
            cfg["data"]["max_train_trajectories"] = 128
            cfg["data"]["max_val_trajectories"] = 16
        elif args.profile == "promote":
            cfg["training"]["max_epochs"] = 6
            cfg["data"]["max_train_trajectories"] = 128
            cfg["data"]["max_val_trajectories"] = 16

        name = variant["name"]
        log_dir = Path("runs") / suite_name / name
        ckpt_dir = Path("checkpoints") / suite_name / name
        cfg["training"]["log_dir"] = str(log_dir)
        cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
        cfg["training"]["verbose_log_file"] = str(log_dir / "train_verbose.log")

        path = cfg_dir / f"{name}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        generated.append(name)
        jobs.append((variant["job"], str(path.relative_to(project_root))))

    with manifest.open("w") as f:
        yaml.safe_dump(
            {
                "suite": suite_name,
                "suite_kind": args.suite_kind,
                "profile": args.profile,
                "generated": generated,
                "notes": suite_spec["notes"],
            },
            f,
            sort_keys=False,
        )

    with jobs_tsv.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["job_name", "config_path"])
        writer.writerows(jobs)

    print(f"Generated {len(generated)} configs under {cfg_dir}")


if __name__ == "__main__":
    main()
