"""Generate and launch the deterministic VDA bug-hunt suite.

The suite is rooted at ``./bug_hunt_vda`` and is intentionally self-contained:

* configs are cloned from ``configs/experiments/tartanair_vda_small_gt.yaml``
* all outputs are rewritten into the suite tree
* baseline runs cover clean and degraded TartanAir v2
* option runs are split into ``phase_a_frozen`` and ``phase_b_unfrozen``
* submission uses the flexible VDA runner, not the GT-only wrapper

This module only touches the generated suite scaffolding and launcher logic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT_DEFAULT = Path.cwd()
SUITE_NAME = "bug_hunt_vda"
SUITE_ROOT_DEFAULT = Path(SUITE_NAME)
BASE_TEMPLATE_DEFAULT = Path("configs/experiments/tartanair_vda_small_gt.yaml")
VDA_RUNNER_DEFAULT = Path("jobs/run_train_vda_small_scratch_distill.sh")
EXPECTED_CHECKPOINT = "checkpoints/metric_video_depth_anything_vits.pth"
VALIDATION_SCOPES = ("clean", "degraded")

RUN_ORDER = (
    "00_clean_ft",
    "01_degraded_ft",
    "02_option1_prefilter",
    "03_option2_refiner",
    "04_option3_quality_gate",
    "05_option4_pre_temporal_adapter",
    "06_option5_aux_recon",
    "07_option6_degradation_conditioned_gate",
)

OPTION_RUNS = RUN_ORDER[2:]

TINY_SUBSET_FULL = {
    "train": 16,
    "val": 4,
    "epochs": 50,
    "warmup_steps": 10,
    "log_images_every_n_steps": 50,
    "step_log_every_n_steps": 10,
}

TINY_SUBSET_SMOKE = {
    "train": 4,
    "val": 2,
    "epochs": 1,
    "warmup_steps": 5,
    "log_images_every_n_steps": 10_000,
    "step_log_every_n_steps": 10,
}

FULL_OPTION_PHASE_A_EPOCHS = 20
FULL_OPTION_PHASE_B_ADDITIONAL_EPOCHS = 50

BASE_MODEL = {
    "type": "video_depth_anything",
    "variant": "small",
    "num_frames": 16,
    "positional_encoding": "ape",
    "checkpoint_path": EXPECTED_CHECKPOINT,
    "strict_checkpoint": False,
    "mode": "offline",
    "stream_max_cache_len": 16,
    "stream_reset_interval": 0,
    "state_gate_enabled": False,
}

BASE_DATA = {
    "dataset": "tartanair_v2",
    "root": "/projects/b5dh/data/tartanair-v2",
    "train_num_frames": 8,
    "val_num_frames": 16,
    "image_size": [392, 392],
    "max_trajectories": None,
    "clip_stride": 8,
    "frame_stride": 1,
    "val_fraction": 0.1,
    "seed": 42,
    "difficulty": "Data_easy",
    "camera": "lcam_front",
    "max_depth": 80.0,
    "envs": None,
    "teacher_cache_dir": None,
}

BASE_TRAINING = {
    "batch_size": 1,
    "learning_rate": 5.0e-5,
    "backbone_lr_multiplier": 0.3,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "precision": "bf16",
    "val_every_n_epochs": 1,
    "selection_metric": "abs_rel",
    "ema_decay": 0.0,
    "num_workers": 8,
    "val_num_workers": 0,
    "gradient_accumulation_steps": 4,
    "compile": False,
    "ddp_find_unused_parameters": True,
    "skip_live_teachers": True,
    "cpu_prefetch": True,
    "cpu_prefetch_batches": 1,
}

BASE_LOSS = {
    "training_target": "metric",
    "ssi_weight": 0.25,
    "temporal_weight": 1.0,
    "temporal_threshold": 0.05,
    "temporal_log_space": False,
    "ssi_mode": "l2",
    "distillation_weight": 0.0,
    "metric_l1_weight": 0.5,
    "metric_log_l1_weight": 1.0,
    "gradient_weight": 0.0,
    "boundary_weight": 0.0,
    "boundary_threshold": 0.05,
    "boundary_dilation": 1,
    "range_prior_weight": 0.0,
    "uncertainty_nll_weight": 0.0,
    "auxiliary": {
        "edge_loss_weight": 0.0,
        "recon_l1_weight": 0.0,
        "recon_ssim_weight": 0.0,
    },
}

BASE_DEGRADATION = {
    "enabled": False,
    "curriculum": False,
}

BASE_VALIDATION = {
    "use_teacher_signals": False,
    "primary_set": "val_clean",
    "sets": [],
}

FIXED_DEGRADED_SETTINGS = {
    "enabled": True,
    "policy": "v2_lowlight",
    "clean_probability": 0.0,
    "curriculum": False,
    "time_mode": "clip_constant",
    "return_params": False,
    "summary": False,
    "drift": {"enabled": False},
    "events": {"enabled": False},
    "processing": {"probability": 0.0},
    "sensor": {"probability": 0.0},
}

FIXED_DEGRADED_SETTINGS_WITH_METADATA = {
    **FIXED_DEGRADED_SETTINGS,
    "return_params": True,
    "summary": True,
}


@dataclass(frozen=True)
class LeafSpec:
    run_name: str
    phase_name: str
    mode: str
    max_epochs: int
    max_train_trajectories: int
    max_val_trajectories: int
    model_overrides: dict[str, Any]
    loss_overrides: dict[str, Any]
    degradation_overrides: dict[str, Any]
    training_overrides: dict[str, Any]
    freeze_prefixes: tuple[str, ...] = ()
    unfreeze_prefixes: tuple[str, ...] = ()
    resume_from: str | None = None
    option_name: str | None = None
    option_role: str | None = None

    @property
    def leaf_dir(self) -> Path:
        if self.phase_name == "single":
            return Path(self.mode)
        return Path(self.phase_name) / self.mode

    @property
    def config_path(self) -> Path:
        return self.leaf_dir / "config.yaml"

    @property
    def launch_path(self) -> Path:
        return self.leaf_dir / "launch.sh"

    @property
    def log_dir(self) -> Path:
        return self.leaf_dir / "logs"

    @property
    def checkpoint_dir(self) -> Path:
        return self.leaf_dir / "checkpoints"

    @property
    def metrics_file(self) -> Path:
        return self.leaf_dir / "metrics.jsonl"

    @property
    def metadata_file(self) -> Path:
        return self.leaf_dir / "run_metadata.json"

    @property
    def config_snapshot_file(self) -> Path:
        return self.leaf_dir / "config_snapshot.yaml"

    @property
    def summary_key(self) -> str:
        return f"{self.run_name}:{self.phase_name}:{self.mode}"


@dataclass(frozen=True)
class RunSpec:
    name: str
    title: str
    stage_type: str
    leaves: tuple[LeafSpec, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = deepcopy(value)
    return dst


def _ensure_suite_dirs(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for run_name in RUN_ORDER:
        (output_root / run_name).mkdir(parents=True, exist_ok=True)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _base_cfg(template_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(template_cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("teachers", [])
    cfg.setdefault("distillation", {})
    cfg.setdefault("training", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("degradation", {})
    cfg.setdefault("validation", {})
    cfg.setdefault("inference", {})

    _deep_update(cfg["model"], deepcopy(BASE_MODEL))
    _deep_update(cfg["data"], deepcopy(BASE_DATA))
    _deep_update(cfg["training"], deepcopy(BASE_TRAINING))
    _deep_update(cfg["loss"], deepcopy(BASE_LOSS))
    _deep_update(cfg["degradation"], deepcopy(BASE_DEGRADATION))
    _deep_update(cfg["validation"], deepcopy(BASE_VALIDATION))
    cfg["teachers"] = []
    cfg["distillation"] = {
        "enabled": False,
        "teachers": [],
        "confidence_weighted": False,
    }
    cfg["inference"] = {"target_fps": 30}
    return cfg


def _set_suite_metadata(
    cfg: dict[str, Any],
    *,
    run_name: str,
    title: str,
    stage_type: str,
    phase_name: str,
    mode: str,
    option_name: str | None,
    option_role: str | None,
    output_root: Path,
    expected_checkpoint: str,
) -> None:
    cfg["suite"] = {
        "name": SUITE_NAME,
        "run_name": run_name,
        "run_title": title,
        "stage_type": stage_type,
        "phase_name": phase_name,
        "mode": mode,
        "option_name": option_name,
        "option_role": option_role,
        "root": str(output_root),
        "validation_scopes": list(VALIDATION_SCOPES),
        "checkpoint_audit": {
            "expected_basename": Path(str(expected_checkpoint)).name,
            "allow_option_missing": bool(option_name),
        },
    }


def _apply_leaf_overrides(
    cfg: dict[str, Any],
    leaf: LeafSpec,
    *,
    run_root: Path,
    output_root: Path,
    title: str,
    stage_type: str,
) -> None:
    training = cfg.setdefault("training", {})
    data = cfg.setdefault("data", {})
    validation = cfg.setdefault("validation", {})
    loss = cfg.setdefault("loss", {})
    degradation = cfg.setdefault("degradation", {})
    model = cfg.setdefault("model", {})
    stage_settings = TINY_SUBSET_SMOKE if leaf.mode == "smoke" else TINY_SUBSET_FULL

    training["max_epochs"] = int(leaf.max_epochs)
    training["warmup_steps"] = int(stage_settings["warmup_steps"])
    training["log_images_every_n_steps"] = int(
        stage_settings["log_images_every_n_steps"]
    )
    training["step_log_every_n_steps"] = int(stage_settings["step_log_every_n_steps"])
    training["validate_before_train"] = True
    training["log_dir"] = str(run_root / leaf.log_dir)
    training["checkpoint_dir"] = str(run_root / leaf.checkpoint_dir)
    training["verbose_log_file"] = str(run_root / leaf.log_dir / "train_verbose.log")
    training["metrics_jsonl_file"] = str(run_root / leaf.metrics_file)
    training["run_metadata_file"] = str(run_root / leaf.metadata_file)
    training["config_snapshot_file"] = str(run_root / leaf.config_snapshot_file)
    training["val_num_workers"] = 0
    training["selection_metric"] = "abs_rel"
    training["ema_decay"] = 0.0
    training["skip_live_teachers"] = True
    training["freeze_prefixes"] = []
    training["unfreeze_prefixes"] = []
    training["freeze_to_option_modules"] = bool(
        leaf.phase_name == "phase_a_frozen" and leaf.option_name
    )
    if training["freeze_to_option_modules"]:
        training["bug_hunt_option_name"] = leaf.option_name
        training["bug_hunt_phase"] = leaf.phase_name
    else:
        training.pop("bug_hunt_option_name", None)
        training.pop("bug_hunt_phase", None)
    training["checkpoint_audit"] = {
        "enabled": True,
        "expected_basename": Path(str(cfg["model"]["checkpoint_path"])).name,
        "allow_option_missing": bool(leaf.option_name),
        "allowed_missing_prefixes": [
            "model.head.motion_modules.",
        ],
    }
    if leaf.resume_from is None:
        training.pop("resume_model_only", None)
        training.pop("resume_additional_epochs", None)
    else:
        training["resume_model_only"] = True
        training["resume_additional_epochs"] = int(leaf.max_epochs)

    data["max_train_trajectories"] = int(leaf.max_train_trajectories)
    data["max_val_trajectories"] = int(leaf.max_val_trajectories)
    data["teacher_cache_dir"] = None
    data["root"] = BASE_DATA["root"]
    data["val_fraction"] = BASE_DATA["val_fraction"]
    data["seed"] = BASE_DATA["seed"]
    data["difficulty"] = BASE_DATA["difficulty"]
    data["camera"] = BASE_DATA["camera"]
    data["train_num_frames"] = BASE_DATA["train_num_frames"]
    data["val_num_frames"] = BASE_DATA["val_num_frames"]
    data["image_size"] = list(BASE_DATA["image_size"])
    data["clip_stride"] = BASE_DATA["clip_stride"]
    data["frame_stride"] = BASE_DATA["frame_stride"]
    data["max_depth"] = BASE_DATA["max_depth"]

    validation["use_teacher_signals"] = False
    validation["primary_set"] = "val_clean" if stage_type == "clean" else "val_degraded"
    degraded_validation_cfg = deepcopy(
        FIXED_DEGRADED_SETTINGS_WITH_METADATA
        if leaf.option_name in {"degradation_gate", "degradation_conditioned_gate"}
        else FIXED_DEGRADED_SETTINGS
    )
    validation["sets"] = [
        {"name": "val_clean", "degradation": {"enabled": False}},
        {"name": "val_degraded", "degradation": degraded_validation_cfg},
    ]

    _deep_update(model, deepcopy(leaf.model_overrides))
    _deep_update(loss, deepcopy(leaf.loss_overrides))
    _deep_update(degradation, deepcopy(leaf.degradation_overrides))
    _deep_update(training, deepcopy(leaf.training_overrides))

    _set_suite_metadata(
        cfg,
        run_name=leaf.run_name,
        title=title,
        stage_type=stage_type,
        phase_name=leaf.phase_name,
        mode=leaf.mode,
        option_name=leaf.option_name,
        option_role=leaf.option_role,
        output_root=output_root,
        expected_checkpoint=str(cfg["model"]["checkpoint_path"]),
    )


def _write_leaf_config(
    *,
    project_root: Path,
    template_cfg: dict[str, Any],
    output_root: Path,
    run_spec: RunSpec,
    leaf: LeafSpec,
) -> Path:
    run_root = output_root / run_spec.name
    cfg = _base_cfg(template_cfg)
    _apply_leaf_overrides(
        cfg,
        leaf,
        run_root=run_root,
        output_root=output_root,
        title=run_spec.title,
        stage_type=run_spec.stage_type,
    )

    cfg_path = run_root / leaf.config_path
    _write_yaml(cfg_path, cfg)

    leaf_manifest = {
        "run_name": run_spec.name,
        "run_title": run_spec.title,
        "stage_type": run_spec.stage_type,
        "phase_name": leaf.phase_name,
        "mode": leaf.mode,
        "config_path": str(cfg_path),
        "launch_script": str(run_root / leaf.launch_path),
        "log_dir": str(run_root / leaf.log_dir),
        "checkpoint_dir": str(run_root / leaf.checkpoint_dir),
        "metrics_file": str(run_root / leaf.metrics_file),
        "metadata_file": str(run_root / leaf.metadata_file),
        "config_snapshot_file": str(run_root / leaf.config_snapshot_file),
        "resume_from": leaf.resume_from,
        "option_name": leaf.option_name,
        "option_role": leaf.option_role,
    }
    _write_yaml(run_root / leaf.leaf_dir / "manifest.yaml", leaf_manifest)

    script = _render_launch_script(
        project_root=project_root,
        config_path=cfg_path,
        job_name=f"vda_bh_{run_spec.name}_{leaf.phase_name}_{leaf.mode}",
        run_root=run_root / leaf.leaf_dir,
        resume_from=leaf.resume_from,
    )
    _write_executable(run_root / leaf.launch_path, script)
    return cfg_path


def _render_launch_script(
    *,
    project_root: Path,
    config_path: Path,
    job_name: str,
    run_root: Path,
    resume_from: str | None,
) -> str:
    resume_export = ""
    if resume_from:
        resume_export = f',RESUME_FROM="{resume_from}"'
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_DIR="${{PROJECT_DIR:-{project_root}}}"
CONFIG="{config_path}"
LOG_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOG_DIR"

sbatch \\
  --job-name="{job_name}" \\
  --output="$LOG_DIR/{job_name}_%j.out" \\
  --error="$LOG_DIR/{job_name}_%j.err" \\
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG="$CONFIG"{resume_export} \\
  "$PROJECT_DIR/{VDA_RUNNER_DEFAULT.as_posix()}"
"""


def _render_root_launcher(mode: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_DIR="${{PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"

exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tools/launch_vda_bug_hunt_suite.py" \\
  --project-root "$PROJECT_DIR" \\
  --output-root "$SCRIPT_DIR" \\
  --mode {mode}
"""


def _render_root_generator() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tools/generate_vda_bug_hunt_suite.py" \\
  --project-root "$PROJECT_DIR" \\
  --output-root "$SCRIPT_DIR"
"""


def _render_phase_leaves(
    *,
    base_name: str,
    title: str,
    stage_type: str,
    option_name: str | None,
    option_role: str | None,
    smoke_only_degraded: bool = False,
    degradation_overrides: dict[str, Any] | None = None,
) -> tuple[LeafSpec, LeafSpec]:
    del smoke_only_degraded
    degradation_overrides = deepcopy(degradation_overrides or {})
    return (
        LeafSpec(
            run_name=base_name,
            phase_name="single",
            mode="smoke",
            max_epochs=TINY_SUBSET_SMOKE["epochs"],
            max_train_trajectories=TINY_SUBSET_SMOKE["train"],
            max_val_trajectories=TINY_SUBSET_SMOKE["val"],
            model_overrides={},
            loss_overrides={},
            degradation_overrides=deepcopy(degradation_overrides),
            training_overrides={},
            option_name=option_name,
            option_role=option_role,
        ),
        LeafSpec(
            run_name=base_name,
            phase_name="single",
            mode="full",
            max_epochs=TINY_SUBSET_FULL["epochs"],
            max_train_trajectories=TINY_SUBSET_FULL["train"],
            max_val_trajectories=TINY_SUBSET_FULL["val"],
            model_overrides={},
            loss_overrides={},
            degradation_overrides=deepcopy(degradation_overrides),
            training_overrides={},
            option_name=option_name,
            option_role=option_role,
        ),
    )


def _run_specs() -> tuple[RunSpec, ...]:
    clean_smoke, clean_full = _render_phase_leaves(
        base_name="00_clean_ft",
        title="VDA clean fine-tune",
        stage_type="clean",
        option_name=None,
        option_role=None,
        degradation_overrides={},
    )
    degraded_smoke, degraded_full = _render_phase_leaves(
        base_name="01_degraded_ft",
        title="VDA degraded fine-tune",
        stage_type="degraded",
        option_name=None,
        option_role=None,
        degradation_overrides=deepcopy(FIXED_DEGRADED_SETTINGS),
    )

    option_runs = []

    def option_phase_leafs(
        *,
        run_name: str,
        title: str,
        option_name: str,
        option_role: str,
        phase_a_model: dict[str, Any],
        phase_a_loss: dict[str, Any],
        phase_a_degradation: dict[str, Any],
        phase_a_freeze: tuple[str, ...] = (),
        phase_a_unfreeze: tuple[str, ...] = (),
        phase_b_model: dict[str, Any] | None = None,
        phase_b_loss: dict[str, Any] | None = None,
        phase_b_degradation: dict[str, Any] | None = None,
        phase_b_freeze: tuple[str, ...] = (),
        phase_b_unfreeze: tuple[str, ...] = (),
    ) -> RunSpec:
        phase_a_smoke = LeafSpec(
            run_name=run_name,
            phase_name="phase_a_frozen",
            mode="smoke",
            max_epochs=1,
            max_train_trajectories=TINY_SUBSET_SMOKE["train"],
            max_val_trajectories=TINY_SUBSET_SMOKE["val"],
            model_overrides=phase_a_model,
            loss_overrides=phase_a_loss,
            degradation_overrides=phase_a_degradation,
            training_overrides={},
            freeze_prefixes=phase_a_freeze,
            unfreeze_prefixes=phase_a_unfreeze,
            option_name=option_name,
            option_role=option_role,
        )
        phase_a_full = LeafSpec(
            run_name=run_name,
            phase_name="phase_a_frozen",
            mode="full",
            max_epochs=FULL_OPTION_PHASE_A_EPOCHS,
            max_train_trajectories=TINY_SUBSET_FULL["train"],
            max_val_trajectories=TINY_SUBSET_FULL["val"],
            model_overrides=phase_a_model,
            loss_overrides=phase_a_loss,
            degradation_overrides=phase_a_degradation,
            training_overrides={},
            freeze_prefixes=phase_a_freeze,
            unfreeze_prefixes=phase_a_unfreeze,
            option_name=option_name,
            option_role=option_role,
        )
        phase_b_model = phase_b_model or phase_a_model
        phase_b_loss = phase_b_loss or phase_a_loss
        phase_b_degradation = phase_b_degradation or phase_a_degradation
        phase_b_smoke = LeafSpec(
            run_name=run_name,
            phase_name="phase_b_unfrozen",
            mode="smoke",
            max_epochs=2,
            max_train_trajectories=TINY_SUBSET_SMOKE["train"],
            max_val_trajectories=TINY_SUBSET_SMOKE["val"],
            model_overrides=phase_b_model,
            loss_overrides=phase_b_loss,
            degradation_overrides=phase_b_degradation,
            training_overrides={},
            freeze_prefixes=phase_b_freeze,
            unfreeze_prefixes=phase_b_unfreeze,
            resume_from="phase_a_frozen/smoke/checkpoints/latest.pt",
            option_name=option_name,
            option_role=option_role,
        )
        phase_b_full = LeafSpec(
            run_name=run_name,
            phase_name="phase_b_unfrozen",
            mode="full",
            max_epochs=FULL_OPTION_PHASE_B_ADDITIONAL_EPOCHS,
            max_train_trajectories=TINY_SUBSET_FULL["train"],
            max_val_trajectories=TINY_SUBSET_FULL["val"],
            model_overrides=phase_b_model,
            loss_overrides=phase_b_loss,
            degradation_overrides=phase_b_degradation,
            training_overrides={},
            freeze_prefixes=phase_b_freeze,
            unfreeze_prefixes=phase_b_unfreeze,
            resume_from="phase_a_frozen/full/checkpoints/latest.pt",
            option_name=option_name,
            option_role=option_role,
        )
        return RunSpec(
            name=run_name,
            title=title,
            stage_type="option",
            leaves=(phase_a_smoke, phase_a_full, phase_b_smoke, phase_b_full),
        )

    option_runs.append(
        option_phase_leafs(
            run_name="02_option1_prefilter",
            title="VDA option 1: front-end prefilter",
            option_name="prefilter",
            option_role="front_adapter_proxy",
            phase_a_model={
                "prefilter_enabled": True,
                "prefilter_type": "stats_guided_front_adapter",
                "front_adapter_hidden": 16,
                "front_adapter_blocks": 2,
                "front_adapter_use_stats_align": True,
                "front_adapter_use_se": True,
            },
            phase_a_loss={},
            phase_a_degradation={
                **deepcopy(FIXED_DEGRADED_SETTINGS),
            },
            phase_a_freeze=("model.pretrained", "model.head"),
            phase_a_unfreeze=("prefilter",),
        )
    )
    option_runs.append(
        option_phase_leafs(
            run_name="03_option2_refiner",
            title="VDA option 2: high-frequency refiner",
            option_name="refiner",
            option_role="refiner_edge",
            phase_a_model={
                "refiner_enabled": True,
                "refiner_channels": 32,
                "refiner_scale": 0.5,
                "refiner_predict_edges": True,
            },
            phase_a_loss={
                "auxiliary": {
                    "edge_loss_weight": 0.2,
                }
            },
            phase_a_degradation=deepcopy(FIXED_DEGRADED_SETTINGS),
        )
    )
    option_runs.append(
        option_phase_leafs(
            run_name="04_option3_quality_gate",
            title="VDA option 3: quality-aware gate",
            option_name="quality_gate",
            option_role="learned_quality_gate",
            phase_a_model={
                "quality_conditioning": {
                    "enabled": True,
                    "dim": 16,
                    "hidden_channels": 16,
                },
            },
            phase_a_loss={},
            phase_a_degradation=deepcopy(FIXED_DEGRADED_SETTINGS),
        )
    )
    option_runs.append(
        option_phase_leafs(
            run_name="05_option4_pre_temporal_adapter",
            title="VDA option 4: pre-temporal stage adapter",
            option_name="pre_temporal_adapter",
            option_role="temporal_adapter",
            phase_a_model={
                "pre_temporal_stage_adapter_enabled": True,
                "pre_temporal_stage_adapter_stages": ["layer3", "layer4"],
                "pre_temporal_stage_adapter_bottleneck_ratio": 4,
            },
            phase_a_loss={},
            phase_a_degradation=deepcopy(FIXED_DEGRADED_SETTINGS),
        )
    )
    option_runs.append(
        option_phase_leafs(
            run_name="06_option5_aux_recon",
            title="VDA option 5: auxiliary reconstruction",
            option_name="aux_recon",
            option_role="aux_reconstruction",
            phase_a_model={
                "aux_reconstruction": {
                    "enabled": True,
                    "channels": 32,
                },
            },
            phase_a_loss={
                "auxiliary": {
                    "recon_l1_weight": 0.5,
                    "recon_ssim_weight": 0.0,
                }
            },
            phase_a_degradation=deepcopy(FIXED_DEGRADED_SETTINGS),
        )
    )
    option_runs.append(
        option_phase_leafs(
            run_name="07_option6_degradation_conditioned_gate",
            title="VDA option 6: degradation-conditioned gate",
            option_name="degradation_conditioned_gate",
            option_role="degradation_conditioned_quality",
            phase_a_model={
                "degradation_conditioning": {
                    "enabled": True,
                    "hidden": 32,
                },
            },
            phase_a_loss={},
            phase_a_degradation=deepcopy(FIXED_DEGRADED_SETTINGS_WITH_METADATA),
        )
    )

    return (
        RunSpec(
            name="00_clean_ft",
            title="VDA clean fine-tune",
            stage_type="clean",
            leaves=(clean_smoke, clean_full),
        ),
        RunSpec(
            name="01_degraded_ft",
            title="VDA degraded fine-tune",
            stage_type="degraded",
            leaves=(degraded_smoke, degraded_full),
        ),
        *option_runs,
    )


def _render_summary_header() -> str:
    return (
        "run\tphase\tmode\tsnapshot\tevent_epoch\tvalidation_scope\tstatus\tcheckpoint\tabs_rel\tdelta1\trmse\t"
        "boundary_f1\tedge_abs_rel\tfdv\tfdv_edge_band\ttgm_metric\tfps\n"
    )


def _validate_required_inputs(project_root: Path, cfg: dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {}) or {}
    required = [
        data_cfg.get("root"),
        cfg.get("model", {}).get("checkpoint_path"),
        project_root / VDA_RUNNER_DEFAULT,
    ]
    distill_cfg = cfg.get("distillation", {}) or {}
    teachers = cfg.get("teachers", []) or []
    if bool(distill_cfg.get("enabled", False)):
        teacher_cache_dir = data_cfg.get("teacher_cache_dir")
        if teacher_cache_dir:
            required.append(teacher_cache_dir)
        for teacher_cfg in teachers:
            if not bool((teacher_cfg or {}).get("enabled", True)):
                continue
            checkpoint_path = (teacher_cfg or {}).get("checkpoint_path")
            if checkpoint_path:
                required.append(checkpoint_path)
    missing = []
    for raw_path in required:
        if not raw_path:
            continue
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.exists():
            missing.append(str(resolved))
    if missing:
        raise FileNotFoundError("Missing required suite inputs:\n" + "\n".join(missing))


def _make_summary_row_spec(
    run: str, phase: str, mode: str, snapshot: str, scope: str
) -> str:
    return (
        f"{run}\t{phase}\t{mode}\t{snapshot}\t\t{scope}\tpending\t\t\t\t\t\t\t\t\t\t\n"
    )


def _summary_metrics_from_event(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validation_sets = event.get("validation_sets")
    if isinstance(validation_sets, dict) and validation_sets:
        return {
            str(name): dict(metrics or {}) for name, metrics in validation_sets.items()
        }

    metrics = dict(event.get("metrics", {}) or {})
    if not metrics:
        return {}

    return {
        str(event.get("primary_validation_set", "val")): metrics,
    }


def _validation_events(metrics_file: Path) -> list[dict[str, Any]]:
    if not metrics_file.is_file():
        return []

    events: list[dict[str, Any]] = []
    for raw_line in metrics_file.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("event") in {"validation", "pretrain_validation"}:
            events.append(event)
    return events


def _latest_validation_event(metrics_file: Path) -> dict[str, Any] | None:
    events = _validation_events(metrics_file)
    return events[-1] if events else None


def _summary_metric_higher_is_better(metric: str) -> bool:
    key = str(metric or "").strip().lower()
    return (
        key.startswith("delta")
        or key.endswith("delta1")
        or key.endswith("delta2")
        or key.endswith("delta3")
        or key
        in {
            "boundary_f1",
            "fps",
        }
    )


def _selection_value_from_event(event: dict[str, Any]) -> tuple[str, float] | None:
    metric = str(event.get("selection_metric") or "abs_rel").strip().lower()
    raw_value = event.get("selection_value")
    if raw_value is not None:
        try:
            return metric, float(raw_value)
        except (TypeError, ValueError):
            return None

    metrics_by_scope = _summary_metrics_from_event(event)
    primary_scope = str(event.get("primary_validation_set") or "")
    primary_metrics = metrics_by_scope.get(primary_scope, {})
    if metric in primary_metrics:
        return metric, float(primary_metrics[metric])

    flat_metrics = event.get("metrics", {}) or {}
    if metric in flat_metrics:
        return metric, float(flat_metrics[metric])
    return None


def _best_validation_event(metrics_file: Path) -> dict[str, Any] | None:
    events = _validation_events(metrics_file)
    if not events:
        return None

    candidates = [
        event for event in events if event.get("event") == "validation"
    ] or events
    best_event: dict[str, Any] | None = None
    best_metric = "abs_rel"
    best_value = float("inf")

    for event in candidates:
        selection = _selection_value_from_event(event)
        if selection is None:
            continue
        metric, value = selection
        if best_event is None:
            best_event = event
            best_metric = metric
            best_value = value
            continue
        if metric != best_metric:
            continue
        if _summary_metric_higher_is_better(metric):
            is_better = value > best_value
        else:
            is_better = value < best_value
        if is_better:
            best_event = event
            best_value = value
    return best_event


def refresh_summary(output_root: Path) -> str:
    manifest = _load_yaml(output_root / "manifest.yaml")
    lines = [_render_summary_header()]
    for run in manifest.get("runs", []):
        run_name = str(run.get("name", ""))
        for leaf in run.get("leaves", []):
            phase_name = str(leaf.get("phase_name", "single"))
            mode = str(leaf.get("mode", "full"))
            leaf_root = _leaf_root(output_root, run_name, phase_name, mode)
            metrics_file = leaf_root / "metrics.jsonl"
            snapshots = {
                "latest": _latest_validation_event(metrics_file),
                "best": _best_validation_event(metrics_file),
            }
            for snapshot_name, event in snapshots.items():
                if event is None:
                    for scope in ("val_clean", "val_degraded"):
                        lines.append(
                            _make_summary_row_spec(
                                run_name, phase_name, mode, snapshot_name, scope
                            )
                        )
                    continue

                checkpoint_name = "best.pt" if snapshot_name == "best" else "latest.pt"
                checkpoint_path = leaf_root / "checkpoints" / checkpoint_name
                if not checkpoint_path.is_file():
                    checkpoint_path = leaf_root / "checkpoints" / "latest.pt"
                checkpoint = str(checkpoint_path)
                metrics_by_scope = _summary_metrics_from_event(event)
                seen_scopes: set[str] = set()
                event_epoch = str(event.get("epoch", ""))
                for scope_name in ("val_clean", "val_degraded"):
                    scope_metrics = metrics_by_scope.get(scope_name)
                    if scope_metrics is None:
                        lines.append(
                            _make_summary_row_spec(
                                run_name, phase_name, mode, snapshot_name, scope_name
                            )
                        )
                        continue
                    seen_scopes.add(scope_name)
                    lines.append(
                        "\t".join(
                            [
                                run_name,
                                phase_name,
                                mode,
                                snapshot_name,
                                event_epoch,
                                scope_name,
                                "ok",
                                checkpoint,
                                str(scope_metrics.get("abs_rel", "")),
                                str(scope_metrics.get("delta1", "")),
                                str(scope_metrics.get("rmse", "")),
                                str(scope_metrics.get("boundary_f1", "")),
                                str(scope_metrics.get("edge_abs_rel", "")),
                                str(scope_metrics.get("fdv", "")),
                                str(scope_metrics.get("fdv_edge_band", "")),
                                str(scope_metrics.get("tgm_metric", "")),
                                str(scope_metrics.get("fps", "")),
                            ]
                        )
                        + "\n"
                    )
                for scope_name, scope_metrics in metrics_by_scope.items():
                    if scope_name in seen_scopes:
                        continue
                    lines.append(
                        "\t".join(
                            [
                                run_name,
                                phase_name,
                                mode,
                                snapshot_name,
                                event_epoch,
                                scope_name,
                                "ok",
                                checkpoint,
                                str(scope_metrics.get("abs_rel", "")),
                                str(scope_metrics.get("delta1", "")),
                                str(scope_metrics.get("rmse", "")),
                                str(scope_metrics.get("boundary_f1", "")),
                                str(scope_metrics.get("edge_abs_rel", "")),
                                str(scope_metrics.get("fdv", "")),
                                str(scope_metrics.get("fdv_edge_band", "")),
                                str(scope_metrics.get("tgm_metric", "")),
                                str(scope_metrics.get("fps", "")),
                            ]
                        )
                        + "\n"
                    )

    summary_text = "".join(lines)
    (output_root / "summary.tsv").write_text(summary_text, encoding="utf-8")
    return summary_text


def generate_suite(
    *,
    project_root: Path,
    output_root: Path,
    template_path: Path,
) -> dict[str, Any]:
    template_cfg = _load_yaml(template_path)
    _validate_required_inputs(project_root, template_cfg)
    base_cfg = _base_cfg(template_cfg)
    suite_checkpoint = str(base_cfg["model"]["checkpoint_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_suite_dirs(output_root)

    suite_manifest: dict[str, Any] = {
        "suite": SUITE_NAME,
        "root": str(output_root),
        "project_root": str(project_root),
        "template": str(template_path),
        "runner": str(project_root / VDA_RUNNER_DEFAULT),
        "expected_checkpoint": EXPECTED_CHECKPOINT,
        "subset": {
            "seed": BASE_DATA["seed"],
            "val_fraction": BASE_DATA["val_fraction"],
            "difficulty": BASE_DATA["difficulty"],
            "camera": BASE_DATA["camera"],
            "train_full": TINY_SUBSET_FULL["train"],
            "val_full": TINY_SUBSET_FULL["val"],
            "train_smoke": TINY_SUBSET_SMOKE["train"],
            "val_smoke": TINY_SUBSET_SMOKE["val"],
        },
        "validation_scopes": list(VALIDATION_SCOPES),
        "runs": [],
        "checkpoint_audit": {
            "expected_checkpoint": suite_checkpoint,
            "strict_load": False,
            "allow_partial_load": True,
        },
    }

    summary_lines = [_render_summary_header()]
    for run_spec in _run_specs():
        run_root = output_root / run_spec.name
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "smoke").mkdir(parents=True, exist_ok=True)
        (run_root / "full").mkdir(parents=True, exist_ok=True)

        run_manifest: dict[str, Any] = {
            "name": run_spec.name,
            "title": run_spec.title,
            "stage_type": run_spec.stage_type,
            "root": str(run_root),
            "leaves": [],
        }
        for leaf in run_spec.leaves:
            cfg_path = _write_leaf_config(
                project_root=project_root,
                template_cfg=template_cfg,
                output_root=output_root,
                run_spec=run_spec,
                leaf=leaf,
            )
            leaf_manifest = {
                "phase_name": leaf.phase_name,
                "mode": leaf.mode,
                "config_path": str(cfg_path),
                "launch_path": str(run_root / leaf.launch_path),
                "log_dir": str(run_root / leaf.log_dir),
                "checkpoint_dir": str(run_root / leaf.checkpoint_dir),
                "resume_from": leaf.resume_from,
                "option_name": leaf.option_name,
                "option_role": leaf.option_role,
            }
            run_manifest["leaves"].append(leaf_manifest)
            summary_lines.append(
                _make_summary_row_spec(
                    run_spec.name, leaf.phase_name, leaf.mode, "latest", "val_clean"
                )
            )
            summary_lines.append(
                _make_summary_row_spec(
                    run_spec.name, leaf.phase_name, leaf.mode, "latest", "val_degraded"
                )
            )
            summary_lines.append(
                _make_summary_row_spec(
                    run_spec.name, leaf.phase_name, leaf.mode, "best", "val_clean"
                )
            )
            summary_lines.append(
                _make_summary_row_spec(
                    run_spec.name, leaf.phase_name, leaf.mode, "best", "val_degraded"
                )
            )

        _write_yaml(run_root / "manifest.yaml", run_manifest)
        suite_manifest["runs"].append(run_manifest)

    _write_yaml(output_root / "manifest.yaml", suite_manifest)
    (output_root / "summary.tsv").write_text("".join(summary_lines), encoding="utf-8")
    _write_executable(output_root / "generate.sh", _render_root_generator())
    _write_executable(output_root / "launch_smoke.sh", _render_root_launcher("smoke"))
    _write_executable(output_root / "launch_full.sh", _render_root_launcher("full"))
    _write_executable(output_root / "launch_all.sh", _render_root_launcher("all"))
    return suite_manifest


def _iter_jobs_for_mode(manifest: dict[str, Any], *, mode: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        for leaf in run.get("leaves", []):
            leaf_mode = str(leaf.get("mode", "full"))
            if leaf_mode != mode:
                continue
            jobs.append({"run": run, "leaf": leaf})

    jobs.sort(
        key=lambda item: (
            RUN_ORDER.index(item["run"]["name"])
            if item["run"]["name"] in RUN_ORDER
            else 999,
            0
            if item["leaf"]["phase_name"] == "single"
            else 1
            if item["leaf"]["phase_name"] == "phase_a_frozen"
            else 2,
        )
    )
    return jobs


def _launch_jobs_for_mode(
    *,
    project_root: Path,
    output_root: Path,
    jobs: list[dict[str, Any]],
    smoke_timeout_s: int,
    full_start_timeout_s: int,
) -> tuple[dict[str, str], dict[str, str]]:
    job_ids: dict[str, str] = {}
    states: dict[str, str] = {}
    for item in jobs:
        run = item["run"]
        leaf = item["leaf"]
        leaf_root = _leaf_root(
            output_root, run["name"], leaf["phase_name"], leaf["mode"]
        )
        cfg_path = Path(leaf["config_path"])
        job_key = run["name"] + ":" + leaf["phase_name"] + ":" + leaf["mode"]
        if _leaf_is_already_complete(leaf_root, cfg_path, mode=str(leaf["mode"])):
            states[job_key] = "SKIPPED_COMPLETE"
            continue
        resume_from = leaf.get("resume_from")
        if resume_from:
            resume_from = str(
                _leaf_root(output_root, run["name"], "phase_a_frozen", leaf["mode"])
                / "checkpoints"
                / "latest.pt"
            )
        job_name = f"vda_bh_{run['name']}_{leaf['phase_name']}_{leaf['mode']}"
        existing_job = _find_existing_job(job_name)
        if existing_job is not None:
            job_id, _existing_state = existing_job
        else:
            job_id = _submit_job(
                project_root=project_root,
                config_path=cfg_path,
                runner_script=project_root / VDA_RUNNER_DEFAULT,
                log_dir=leaf_root / "slurm_logs",
                job_name=job_name,
                resume_from=resume_from,
            )
        job_ids[job_key] = job_id

        wait_timeout = (
            smoke_timeout_s if leaf["mode"] == "smoke" else full_start_timeout_s
        )
        allowed = {"COMPLETED"} if leaf["mode"] == "smoke" else {"RUNNING", "COMPLETED"}
        state = _wait_for_state(job_id, allowed, timeout_s=wait_timeout)
        states[job_key] = state

        if leaf["mode"] == "smoke":
            _assert_smoke_outputs(leaf_root)
        _assert_run_outputs_inside_root(leaf_root, cfg_path)

    return job_ids, states


def _extract_job_id(sbatch_output: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {sbatch_output!r}")
    return match.group(1)


def _run_cmd(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _submit_job(
    *,
    project_root: Path,
    config_path: Path,
    runner_script: Path,
    log_dir: Path,
    job_name: str,
    resume_from: str | None = None,
    dependency_afterok: str | None = None,
) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / f"{job_name}_%j.out"
    err_log = log_dir / f"{job_name}_%j.err"
    export_bits = [f"ALL,PROJECT_DIR={project_root}", f"CONFIG={config_path}"]
    if resume_from:
        export_bits.append(f"RESUME_FROM={resume_from}")
    output = _run_cmd(
        [
            "sbatch",
            "--job-name",
            job_name,
            "--output",
            str(out_log),
            "--error",
            str(err_log),
            *(
                []
                if not dependency_afterok
                else ["--dependency", f"afterok:{dependency_afterok}"]
            ),
            "--export",
            ",".join(export_bits),
            str(runner_script),
        ],
        cwd=project_root,
    )
    return _extract_job_id(output)


def _slurm_state(job_id: str) -> str:
    try:
        out = _run_cmd(["squeue", "-h", "-j", str(job_id), "-o", "%T"])
        if out:
            return out.splitlines()[0].strip()
    except Exception:
        pass
    try:
        out = _run_cmd(["sacct", "-n", "-P", "-j", str(job_id), "--format=State"])
        states = [
            line.strip().split("|")[0] for line in out.splitlines() if line.strip()
        ]
        if states:
            return states[0]
    except Exception:
        pass
    return "UNKNOWN"


def _find_existing_job(job_name: str) -> tuple[str, str] | None:
    try:
        out = _run_cmd(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j|%T"]
        )
        matches: list[tuple[int, str, str]] = []
        for line in out.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 3:
                continue
            job_id, seen_name, state = parts
            if seen_name != job_name or not job_id.isdigit():
                continue
            matches.append((int(job_id), job_id, state))
        if matches:
            matches.sort()
            _job_num, job_id, state = matches[-1]
            return job_id, state
    except Exception:
        pass

    try:
        out = _run_cmd(
            [
                "sacct",
                "-n",
                "-P",
                "-u",
                os.environ.get("USER", ""),
                "--format=JobIDRaw,JobName,State",
            ]
        )
        matches = []
        for line in out.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3:
                continue
            job_id, seen_name, state = parts[:3]
            if seen_name != job_name or not job_id.isdigit():
                continue
            matches.append((int(job_id), job_id, state))
        if matches:
            matches.sort()
            _job_num, job_id, state = matches[-1]
            return job_id, state
    except Exception:
        pass

    return None


def _enqueue_full_jobs_with_dependencies(
    *,
    project_root: Path,
    output_root: Path,
    jobs: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    job_ids: dict[str, str] = {}
    states: dict[str, str] = {}
    leaf_complete: dict[str, bool] = {}

    for item in jobs:
        run = item["run"]
        leaf = item["leaf"]
        job_key = run["name"] + ":" + leaf["phase_name"] + ":" + leaf["mode"]
        leaf_root = _leaf_root(
            output_root, run["name"], leaf["phase_name"], leaf["mode"]
        )
        cfg_path = Path(leaf["config_path"])
        is_complete = _leaf_is_already_complete(
            leaf_root, cfg_path, mode=str(leaf["mode"])
        )
        leaf_complete[job_key] = is_complete
        if is_complete:
            states[job_key] = "SKIPPED_COMPLETE"
            continue

        _assert_run_outputs_inside_root(leaf_root, cfg_path)
        job_name = f"vda_bh_{run['name']}_{leaf['phase_name']}_{leaf['mode']}"

        existing_job = _find_existing_job(job_name)
        if existing_job is not None:
            existing_job_id, existing_state = existing_job
            existing_state_upper = existing_state.upper()
            if existing_state_upper in {
                "PENDING",
                "CONFIGURING",
                "RUNNING",
                "COMPLETING",
            }:
                job_ids[job_key] = existing_job_id
                states[job_key] = f"REUSED_{existing_state_upper}"
                continue

        resume_from = leaf.get("resume_from")
        if resume_from:
            resume_from = str(
                _leaf_root(output_root, run["name"], "phase_a_frozen", leaf["mode"])
                / "checkpoints"
                / "latest.pt"
            )

        dependency_afterok: str | None = None
        if str(leaf["phase_name"]) == "phase_b_unfrozen":
            phase_a_key = run["name"] + ":phase_a_frozen:" + leaf["mode"]
            if not leaf_complete.get(phase_a_key, False):
                dependency_afterok = job_ids.get(phase_a_key)
                if dependency_afterok is None:
                    phase_a_job_name = (
                        f"vda_bh_{run['name']}_phase_a_frozen_{leaf['mode']}"
                    )
                    existing_phase_a = _find_existing_job(phase_a_job_name)
                    if existing_phase_a is not None:
                        dependency_afterok = existing_phase_a[0]
                if dependency_afterok is None:
                    raise RuntimeError(
                        f"Could not resolve phase-a dependency for {job_key}; phase_a_frozen full job must exist first."
                    )

        job_id = _submit_job(
            project_root=project_root,
            config_path=cfg_path,
            runner_script=project_root / VDA_RUNNER_DEFAULT,
            log_dir=leaf_root / "slurm_logs",
            job_name=job_name,
            resume_from=resume_from,
            dependency_afterok=dependency_afterok,
        )
        job_ids[job_key] = job_id
        states[job_key] = _slurm_state(job_id)

    return job_ids, states


def _wait_for_state(
    job_id: str, allowed: set[str], *, timeout_s: int, poll_s: int = 20
) -> str:
    deadline = time.time() + timeout_s
    last_state = "PENDING"
    while time.time() < deadline:
        last_state = _slurm_state(job_id)
        normalized = last_state.upper()
        if normalized in allowed:
            return normalized
        time.sleep(poll_s)
    raise TimeoutError(
        f"Timed out waiting for job {job_id} to reach one of {sorted(allowed)}; last state={last_state}"
    )


def _leaf_root(output_root: Path, run_name: str, phase_name: str, mode: str) -> Path:
    if phase_name == "single":
        return output_root / run_name / mode
    return output_root / run_name / phase_name / mode


def _assert_smoke_outputs(leaf_root: Path) -> None:
    metrics_file = leaf_root / "metrics.jsonl"
    latest_ckpt = leaf_root / "checkpoints" / "latest.pt"
    if not latest_ckpt.is_file():
        raise RuntimeError(f"Smoke run did not save latest checkpoint: {latest_ckpt}")
    if not metrics_file.is_file():
        raise RuntimeError(f"Smoke run did not write metrics.jsonl: {metrics_file}")
    metrics_text = metrics_file.read_text(encoding="utf-8")
    if '"event": "validation"' not in metrics_text:
        raise RuntimeError(f"Smoke run did not reach validation: {metrics_file}")


def _assert_run_outputs_inside_root(leaf_root: Path, cfg_path: Path) -> None:
    cfg = _load_yaml(cfg_path)
    train_cfg = cfg.get("training", {}) or {}
    keys = (
        "log_dir",
        "checkpoint_dir",
        "verbose_log_file",
        "metrics_jsonl_file",
        "run_metadata_file",
        "config_snapshot_file",
    )
    for key in keys:
        value = train_cfg.get(key)
        if not value:
            continue
        path = Path(str(value)).resolve()
        if leaf_root.resolve() not in path.parents and path != leaf_root.resolve():
            raise RuntimeError(
                f"Generated path for {key} escapes run root {leaf_root}: {path}"
            )


def _metrics_file_has_event(metrics_file: Path, event_name: str) -> bool:
    if not metrics_file.is_file():
        return False
    for raw_line in metrics_file.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == event_name:
            return True
    return False


def _leaf_is_already_complete(leaf_root: Path, cfg_path: Path, *, mode: str) -> bool:
    try:
        _assert_run_outputs_inside_root(leaf_root, cfg_path)
    except Exception:
        return False

    if mode == "smoke":
        try:
            _assert_smoke_outputs(leaf_root)
        except Exception:
            return False
        return True

    latest_ckpt = leaf_root / "checkpoints" / "latest.pt"
    metrics_file = leaf_root / "metrics.jsonl"
    return latest_ckpt.is_file() and _metrics_file_has_event(
        metrics_file, "run_complete"
    )


def launch_suite(
    *,
    project_root: Path,
    output_root: Path,
    mode: str = "all",
    smoke_timeout_s: int = 6 * 3600,
    full_start_timeout_s: int = 15 * 60,
) -> dict[str, Any]:
    manifest = _load_yaml(output_root / "manifest.yaml")
    job_ids: dict[str, str] = {}
    states: dict[str, str] = {}

    modes = ("smoke", "full") if mode == "all" else (mode,)
    for active_mode in modes:
        active_jobs = _iter_jobs_for_mode(manifest, mode=active_mode)
        if active_mode == "full":
            mode_job_ids, mode_states = _enqueue_full_jobs_with_dependencies(
                project_root=project_root,
                output_root=output_root,
                jobs=active_jobs,
            )
        else:
            mode_job_ids, mode_states = _launch_jobs_for_mode(
                project_root=project_root,
                output_root=output_root,
                jobs=active_jobs,
                smoke_timeout_s=smoke_timeout_s,
                full_start_timeout_s=full_start_timeout_s,
            )
        job_ids.update(mode_job_ids)
        states.update(mode_states)

    launch_report = {
        "mode": mode,
        "jobs": job_ids,
        "states": states,
        "output_root": str(output_root),
    }
    (output_root / "launch_report.json").write_text(
        json.dumps(launch_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    refresh_summary(output_root)
    return launch_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or launch the VDA bug-hunt suite."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT_DEFAULT
    )
    generate_parser.add_argument("--output-root", type=Path, default=SUITE_ROOT_DEFAULT)
    generate_parser.add_argument("--template", type=Path, default=BASE_TEMPLATE_DEFAULT)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT_DEFAULT
    )
    launch_parser.add_argument("--output-root", type=Path, default=SUITE_ROOT_DEFAULT)
    launch_parser.add_argument(
        "--mode", choices=("smoke", "full", "all"), default="all"
    )
    launch_parser.add_argument("--smoke-timeout-s", type=int, default=6 * 3600)
    launch_parser.add_argument("--full-start-timeout-s", type=int, default=15 * 60)

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_suite(
            project_root=args.project_root.resolve(),
            output_root=args.output_root.resolve(),
            template_path=(args.project_root / args.template).resolve()
            if not args.template.is_absolute()
            else args.template.resolve(),
        )
        return 0
    if args.command == "launch":
        launch_suite(
            project_root=args.project_root.resolve(),
            output_root=args.output_root.resolve(),
            mode=args.mode,
            smoke_timeout_s=args.smoke_timeout_s,
            full_start_timeout_s=args.full_start_timeout_s,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
