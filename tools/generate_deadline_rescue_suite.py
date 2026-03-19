#!/usr/bin/env python3
"""Generate the deadline-driven degraded streaming rescue sweep."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import yaml


PROJECT_ROOT = Path("/projects/b5dh/FocusMamba")
SUITE_NAME = "deadline_rescue_sweep_20260309"
CFG_DIR = PROJECT_ROOT / "configs" / "experiments" / "generated" / SUITE_NAME
RUN_ROOT = PROJECT_ROOT / "runs" / SUITE_NAME
CKPT_ROOT = PROJECT_ROOT / "checkpoints" / SUITE_NAME
JOBS_TSV = CFG_DIR / "jobs.tsv"
MANIFEST = CFG_DIR / "manifest.yaml"


def deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


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
        "prefilter_kernel_size": 5,
        "prefilter_sigma": 1.0,
        "prefilter_denoise_init": 0.20,
        "prefilter_sharpen_init": 0.10,
        "prefilter_learnable": True,
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
        "teacher_cache_dir": None,
    },
    "teachers": [],
    "distillation": {
        "enabled": False,
        "teachers": [],
        "confidence_weighted": False,
    },
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
        "max_epochs": 2,
        "batch_size": 1,
        "learning_rate": 5.0e-05,
        "backbone_lr_multiplier": 0.3,
        "weight_decay": 0.01,
        "warmup_steps": 100,
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
    "validation": {
        "use_teacher_signals": False,
    },
    "loss": {
        "training_target": "metric",
        "ssi_weight": 0.25,
        "temporal_weight": 1.0,
        "temporal_threshold": 0.05,
        "temporal_log_space": False,
        "ssi_mode": "l2",
        "distillation_weight": 0.0,
        "metric_l1_weight": 0.5,
        "metric_log_l1_weight": 1.0,
    },
}

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


VARIANTS = [
    {
        "name": "00_stream_clean_screen",
        "job": "rCln",
        "overrides": {
            "degradation": {"enabled": False},
        },
    },
    {"name": "01_plain_degraded", "job": "rDeg", "overrides": {}},
    {
        "name": "02_reset4",
        "job": "rR04",
        "overrides": {"model": {"stream_reset_interval": 4}},
    },
    {
        "name": "03_reset8",
        "job": "rR08",
        "overrides": {"model": {"stream_reset_interval": 8}},
    },
    {
        "name": "04_shortmem4",
        "job": "rK04",
        "overrides": {"model": {"stream_max_cache_len": 4}},
    },
    {
        "name": "05_shortmem8",
        "job": "rK08",
        "overrides": {"model": {"stream_max_cache_len": 8}},
    },
    {
        "name": "06_gate_all",
        "job": "rGat",
        "overrides": {"model": {"state_gate_enabled": True}},
    },
    {
        "name": "07_gate_last",
        "job": "rGLs",
        "overrides": {
            "model": {
                "state_gate_enabled": True,
                "state_gate_stage_mask": [False, False, False, True],
            }
        },
    },
    {
        "name": "08_gate_temporal_head",
        "job": "rGTh",
        "overrides": {
            "model": {"state_gate_enabled": True},
            "training": {
                "freeze_backbone": True,
                "unfreeze_prefixes": [
                    "model.head.motion_modules",
                    "model.head.stage_gates",
                ],
            },
        },
    },
    {
        "name": "09_prefilter_fast_fixed",
        "job": "rPff",
        "overrides": {
            "model": {
                "prefilter_enabled": True,
                "prefilter_type": "fast_classical",
                "prefilter_learnable": False,
                "prefilter_denoise_init": 0.25,
                "prefilter_sharpen_init": 0.10,
            }
        },
    },
    {
        "name": "10_stats_align_fixed",
        "job": "rSta",
        "overrides": {
            "model": {
                "prefilter_enabled": True,
                "prefilter_type": "stats_align",
                "prefilter_learnable": False,
            }
        },
    },
    {
        "name": "11_affine_norm",
        "job": "rAff",
        "overrides": {
            "model": {"prefilter_enabled": True, "prefilter_type": "learned_affine"}
        },
    },
    {
        "name": "12_depthwise_norm",
        "job": "rDpw",
        "overrides": {
            "model": {"prefilter_enabled": True, "prefilter_type": "depthwise"}
        },
    },
    {
        "name": "13_gate_prefilter_fast",
        "job": "rGPf",
        "overrides": {
            "model": {
                "state_gate_enabled": True,
                "prefilter_enabled": True,
                "prefilter_type": "fast_classical",
                "prefilter_learnable": False,
                "prefilter_denoise_init": 0.25,
                "prefilter_sharpen_init": 0.10,
            }
        },
    },
    {
        "name": "14_gate_affine_norm",
        "job": "rGAf",
        "overrides": {
            "model": {
                "state_gate_enabled": True,
                "prefilter_enabled": True,
                "prefilter_type": "learned_affine",
            }
        },
    },
    {
        "name": "15_curriculum_ramp",
        "job": "rRmp",
        "overrides": {
            "degradation": {
                "curriculum": {
                    "warmup_epochs": 0,
                    "max_severity_epoch": 2,
                    "schedule": "linear",
                    "warmup_start_scale": 0.0,
                    "min_scale": 0.1,
                    "max_scale": 1.0,
                }
            }
        },
    },
    {
        "name": "16_curriculum_moderate",
        "job": "rMod",
        "overrides": {
            "degradation": {
                "curriculum": {
                    "warmup_epochs": 1,
                    "max_severity_epoch": 2,
                    "schedule": "linear",
                    "warmup_start_scale": 0.5,
                    "min_scale": 0.5,
                    "max_scale": 1.0,
                }
            }
        },
    },
    {
        "name": "17_curriculum_mixed",
        "job": "rMix",
        "overrides": {
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
        },
    },
    {
        "name": "18_teacher_bugfix",
        "job": "rTBg",
        "overrides": {
            "data": {
                "teacher_cache_dir": "/projects/b5dh/data/teacher_cache/tartanair_v2"
            },
            "teachers": TEACHERS,
            "distillation": {
                "enabled": True,
                "target_mode": "metric",
                "strategy": "legacy",
                "teachers": [{"name": t["name"], "weight": 1.0} for t in TEACHERS],
                "confidence_weighted": True,
                "lambda_si": 0.5,
            },
            "loss": {"distillation_weight": 0.15},
        },
    },
    {
        "name": "19_teacher_aggregate",
        "job": "rTAg",
        "overrides": {
            "data": {
                "teacher_cache_dir": "/projects/b5dh/data/teacher_cache/tartanair_v2"
            },
            "teachers": TEACHERS,
            "distillation": {
                "enabled": True,
                "target_mode": "metric",
                "strategy": "calibrated_aggregate",
                "teachers": [{"name": t["name"], "weight": 1.0} for t in TEACHERS],
                "confidence_weighted": True,
                "lambda_si": 0.5,
                "calibration": {
                    "enabled": True,
                    "min_depth": 0.1,
                    "min_valid_pixels": 64,
                    "use_gt": True,
                },
                "aggregate": {"weight": 1.0, "beta": 6.0},
            },
            "loss": {"distillation_weight": 0.15},
        },
    },
    {
        "name": "20_teacher_hetero",
        "job": "rTHt",
        "overrides": {
            "data": {
                "teacher_cache_dir": "/projects/b5dh/data/teacher_cache/tartanair_v2"
            },
            "teachers": TEACHERS,
            "distillation": {
                "enabled": True,
                "target_mode": "metric",
                "strategy": "heterogeneous_mtkd",
                "teachers": [{"name": t["name"], "weight": 1.0} for t in TEACHERS],
                "confidence_weighted": True,
                "lambda_si": 0.5,
                "calibration": {
                    "enabled": True,
                    "min_depth": 0.1,
                    "min_valid_pixels": 64,
                    "use_gt": True,
                },
                "aggregate": {"weight": 0.5, "beta": 6.0},
                "auxiliary": {
                    "vda_weight": 0.3,
                    "da3_weight": 0.1,
                    "depth_pro_weight": 0.1,
                    "vda_temporal_weight": 0.5,
                    "structure_weight": 0.5,
                    "gradient_scales": 1,
                },
            },
            "loss": {"distillation_weight": 0.15},
        },
    },
    {
        "name": "21_teacher_hetero_gate",
        "job": "rTGt",
        "overrides": {
            "model": {"state_gate_enabled": True},
            "data": {
                "teacher_cache_dir": "/projects/b5dh/data/teacher_cache/tartanair_v2"
            },
            "teachers": TEACHERS,
            "distillation": {
                "enabled": True,
                "target_mode": "metric",
                "strategy": "heterogeneous_mtkd",
                "teachers": [{"name": t["name"], "weight": 1.0} for t in TEACHERS],
                "confidence_weighted": True,
                "lambda_si": 0.5,
                "calibration": {
                    "enabled": True,
                    "min_depth": 0.1,
                    "min_valid_pixels": 64,
                    "use_gt": True,
                },
                "aggregate": {"weight": 0.5, "beta": 6.0},
                "auxiliary": {
                    "vda_weight": 0.3,
                    "da3_weight": 0.1,
                    "depth_pro_weight": 0.1,
                    "vda_temporal_weight": 0.5,
                    "structure_weight": 0.5,
                    "gradient_scales": 1,
                },
            },
            "loss": {"distillation_weight": 0.15},
        },
    },
    {
        "name": "22_teacher_single_da3",
        "job": "rTSg",
        "overrides": {
            "data": {
                "teacher_cache_dir": "/projects/b5dh/data/teacher_cache/tartanair_v2"
            },
            "teachers": [TEACHERS[1]],
            "distillation": {
                "enabled": True,
                "target_mode": "metric",
                "strategy": "calibrated_aggregate",
                "teachers": [{"name": "depth_anything_v3", "weight": 1.0}],
                "confidence_weighted": True,
                "lambda_si": 0.5,
                "calibration": {
                    "enabled": True,
                    "min_depth": 0.1,
                    "min_valid_pixels": 64,
                    "use_gt": True,
                },
                "aggregate": {"weight": 1.0, "beta": 6.0},
            },
            "loss": {"distillation_weight": 0.15},
        },
    },
    {
        "name": "23_feature_align",
        "job": "rFtr",
        "overrides": {
            "loss": {
                "auxiliary": {
                    "feature_alignment_weight": 0.05,
                }
            }
        },
    },
    {
        "name": "24_feature_align_gate",
        "job": "rFGt",
        "overrides": {
            "model": {"state_gate_enabled": True},
            "loss": {
                "auxiliary": {
                    "feature_alignment_weight": 0.05,
                }
            },
        },
    },
    {
        "name": "25_temporal_smooth",
        "job": "rTmp",
        "overrides": {
            "loss": {
                "temporal_weight": 2.5,
                "temporal_threshold": 0.03,
            }
        },
    },
    {
        "name": "26_depth_consistency",
        "job": "rDcs",
        "overrides": {
            "loss": {
                "auxiliary": {
                    "clean_depth_consistency_weight": 0.05,
                    "clean_depth_consistency_mode": "log_l1",
                }
            }
        },
    },
]


def main() -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    generated = []
    jobs = []

    for variant in VARIANTS:
        name = variant["name"]
        cfg = deepcopy(BASE)
        deep_update(cfg, deepcopy(variant["overrides"]))

        log_dir = Path("runs") / SUITE_NAME / name
        ckpt_dir = Path("checkpoints") / SUITE_NAME / name
        cfg["training"]["log_dir"] = str(log_dir)
        cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
        cfg["training"]["verbose_log_file"] = str(log_dir / "train_verbose.log")

        path = CFG_DIR / f"{name}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        generated.append(name)
        jobs.append((variant["job"], str(path.relative_to(PROJECT_ROOT))))

    manifest = {
        "suite": SUITE_NAME,
        "generated": generated,
        "notes": [
            "Fast-loop deadline-driven rescue sweep for degraded causal streaming VDA-S.",
            "Uses train_num_frames=8, val_num_frames=16, val_num_workers=0, and trajectory subsets for screening.",
            "Covers baseline memory heuristics, input normalizers, curricula, distillation, and clean/degraded auxiliary alignment.",
        ],
    }
    with MANIFEST.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    with JOBS_TSV.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["job_name", "config_path"])
        writer.writerows(jobs)

    print(f"Generated {len(generated)} configs in {CFG_DIR}")


if __name__ == "__main__":
    main()
