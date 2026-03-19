"""Generation and launch helpers for the FocusMamba six-option suite."""

from __future__ import annotations

import argparse
import json
import re
import stat
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SUITE_RUNS: tuple[str, ...] = (
    "baseline",
    "option1_prefilter",
    "option2_refiner",
    "option3_quality_gate",
    "option4_temporal_mamba",
    "option5_aux_recon",
    "option6_degradation_film",
)

OPTION_RUNS: tuple[str, ...] = tuple(run for run in SUITE_RUNS if run != "baseline")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def _standardize_baseline(
    cfg: dict[str, Any], *, run_root: Path, smoke: bool
) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("model", {})
    out.setdefault("data", {})
    out.setdefault("training", {})
    out.setdefault("degradation", {})
    out.setdefault("validation", {})
    out.setdefault("loss", {})

    out["degradation"].update(
        {
            "enabled": True,
            "policy": "v2_lowlight",
            "return_params": True,
            "summary": True,
            "time_mode": "drift_events",
            "clean_probability": 0.10,
            "curriculum": False,
            "drift": {
                "enabled": True,
                "exposure_strength": 0.35,
                "noise_strength": 0.35,
                "blur_strength": 0.30,
            },
            "events": {
                "enabled": True,
                "exposure_jump_probability": 0.3,
                "exposure_jump_magnitude": 1.0,
                "blur_spike_probability": 0.3,
                "blur_spike_factor": 2.5,
            },
            "processing": {"probability": 0.25},
            "sensor": {
                "probability": 0.10,
                "line_jitter_max_shift": 2,
                "hot_dead_pixel_probability": 0.001,
            },
        }
    )
    out["validation"]["robodepth"] = {"enabled": False}
    out["training"]["selection_metric"] = "abs_rel"

    if smoke:
        out["training"]["max_epochs"] = 1
        out["training"]["val_every_n_epochs"] = 1
        out["training"]["log_images_every_n_steps"] = max(
            int(out["training"].get("log_images_every_n_steps", 1000)),
            10_000,
        )
        out["data"]["max_train_trajectories"] = 4
        out["data"]["max_val_trajectories"] = 2
        out["data"]["clip_stride"] = max(int(out["data"].get("clip_stride", 8)), 32)
        out["data"]["image_size"] = [256, 256]

    mode_root = run_root / ("smoke" if smoke else "full")
    log_dir = mode_root / "logs"
    checkpoint_dir = mode_root / "checkpoints"
    out["training"]["log_dir"] = str(log_dir)
    out["training"]["checkpoint_dir"] = str(checkpoint_dir)
    out["training"]["verbose_log_file"] = str(log_dir / "train_verbose.log")
    out["training"]["metrics_jsonl_file"] = str(mode_root / "metrics.jsonl")
    out["training"]["run_metadata_file"] = str(mode_root / "run_metadata.json")
    out["training"]["config_snapshot_file"] = str(mode_root / "config_snapshot.yaml")
    return out


def _apply_option_delta(run_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg)
    model_cfg = out.setdefault("model", {})
    loss_cfg = out.setdefault("loss", {})
    aux_cfg = loss_cfg.setdefault("auxiliary", {})

    if run_name == "option1_prefilter":
        model_cfg.update(
            {
                "prefilter_enabled": True,
                "prefilter_type": "stats_guided_front_adapter",
                "prefilter_learnable": True,
                "front_adapter_hidden": 16,
                "front_adapter_blocks": 2,
            }
        )
    elif run_name == "option2_refiner":
        model_cfg.update(
            {
                "refiner_enabled": True,
                "refiner_channels": 32,
                "refiner_scale": 0.5,
                "refiner_predict_edges": True,
            }
        )
        aux_cfg["edge_loss_weight"] = 0.2
    elif run_name == "option3_quality_gate":
        model_cfg["quality_conditioning"] = {
            "enabled": True,
            "dim": 16,
            "hidden_channels": 16,
        }
    elif run_name == "option4_temporal_mamba":
        model_cfg.update(
            {
                "temporal_module_type": "mamba",
                "temporal_mamba_d_state": 16,
                "temporal_mamba_d_conv": 4,
                "temporal_mamba_expand": 2,
            }
        )
    elif run_name == "option5_aux_recon":
        model_cfg["aux_reconstruction"] = {
            "enabled": True,
            "channels": 32,
        }
        aux_cfg["recon_l1_weight"] = 0.5
        aux_cfg["recon_ssim_weight"] = 0.0
    elif run_name == "option6_degradation_film":
        model_cfg["degradation_conditioning"] = {
            "enabled": True,
            "method": "film",
            "target": "backbone_scan_weights",
            "dim": 32,
        }
    return out


def _validate_required_paths(cfg: dict[str, Any], *, project_root: Path) -> None:
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    distill_cfg = cfg.get("distillation", {}) or {}
    teacher_list = cfg.get("teachers", []) or []

    required_paths = [
        Path(data_cfg["root"]) if data_cfg.get("root") else None,
    ]
    if bool(train_cfg.get("skip_live_teachers", False)) and bool(
        distill_cfg.get("enabled", False)
    ):
        required_paths.append(
            Path(data_cfg["teacher_cache_dir"])
            if data_cfg.get("teacher_cache_dir")
            else None
        )
    for teacher in teacher_list:
        if teacher.get("enabled", True) and teacher.get("checkpoint_path"):
            required_paths.append(Path(str(teacher["checkpoint_path"])))

    missing = []
    for path in required_paths:
        if path is None:
            continue
        resolved = path if path.is_absolute() else (project_root / path)
        if not resolved.exists():
            missing.append(str(resolved))
    if missing:
        raise FileNotFoundError("Missing required suite inputs:\n" + "\n".join(missing))


def _write_launch_script(
    path: Path, *, config_path: Path, runner_script: Path, job_name: str, mode: str
) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_DIR="${{PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
CFG="$SCRIPT_DIR/{config_path.name}"
LOG_DIR="$SCRIPT_DIR/{mode}/slurm_logs"
mkdir -p "$LOG_DIR"
sbatch \\
  --job-name="{job_name}" \\
  --output="$LOG_DIR/{job_name}_%j.out" \\
  --error="$LOG_DIR/{job_name}_%j.err" \\
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG="$CFG" \\
  "{runner_script}"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def generate_suite(
    *,
    project_root: Path,
    output_root: Path,
    full_template: Path,
    smoke_template: Path,
) -> dict[str, Any]:
    full_base = _load_yaml(full_template)
    smoke_base = _load_yaml(smoke_template)

    suite_manifest = {
        "suite": "focusmamba_six_runs",
        "root": str(output_root),
        "validation_dataset": "tartanair_v2",
        "selection_metric": "abs_rel",
        "runs": [],
    }

    output_root.mkdir(parents=True, exist_ok=True)

    for run_name in SUITE_RUNS:
        run_root = output_root / run_name
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "smoke").mkdir(exist_ok=True)
        (run_root / "full").mkdir(exist_ok=True)

        full_cfg = _apply_option_delta(
            run_name, _standardize_baseline(full_base, run_root=run_root, smoke=False)
        )
        smoke_cfg = _apply_option_delta(
            run_name, _standardize_baseline(smoke_base, run_root=run_root, smoke=True)
        )

        _validate_required_paths(full_cfg, project_root=project_root)

        full_cfg_path = run_root / "config.full.yaml"
        smoke_cfg_path = run_root / "config.smoke.yaml"
        _write_yaml(full_cfg_path, full_cfg)
        _write_yaml(smoke_cfg_path, smoke_cfg)

        full_runner = project_root / "jobs" / "run_train_focusmamba_v2_distill_50ep.sh"
        smoke_runner = (
            project_root / "jobs" / "run_train_focusmamba_v2_smoke_distill.sh"
        )
        _write_launch_script(
            run_root / "launch_full.sh",
            config_path=full_cfg_path,
            runner_script=full_runner,
            job_name=f"fm6_{run_name}_full",
            mode="full",
        )
        _write_launch_script(
            run_root / "launch_smoke.sh",
            config_path=smoke_cfg_path,
            runner_script=smoke_runner,
            job_name=f"fm6_{run_name}_smoke",
            mode="smoke",
        )

        run_manifest = {
            "name": run_name,
            "root": str(run_root),
            "full_config": str(full_cfg_path),
            "smoke_config": str(smoke_cfg_path),
            "full_runner": str(full_runner),
            "smoke_runner": str(smoke_runner),
            "selection_metric": "abs_rel",
            "validation_dataset": "tartanair_v2",
        }
        _write_yaml(run_root / "manifest.yaml", run_manifest)
        suite_manifest["runs"].append(run_manifest)

    _write_yaml(output_root / "manifest.yaml", suite_manifest)
    summary_tsv = output_root / "summary.tsv"
    if not summary_tsv.exists():
        summary_tsv.write_text(
            "run\tmode\tabs_rel\tdelta1\tboundary_f1\tedge_abs_rel\tfdv\tfdv_edge_band\ttgm_metric\tfps\tstatus\n",
            encoding="utf-8",
        )
    return suite_manifest


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
) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / f"{job_name}_%j.out"
    err_log = log_dir / f"{job_name}_%j.err"
    output = _run_cmd(
        [
            "sbatch",
            "--job-name",
            job_name,
            "--output",
            str(out_log),
            "--error",
            str(err_log),
            "--export",
            f"ALL,PROJECT_DIR={project_root},CONFIG={config_path}",
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


def _assert_smoke_outputs(run_root: Path) -> None:
    metrics_file = run_root / "smoke" / "metrics.jsonl"
    latest_ckpt = run_root / "smoke" / "checkpoints" / "latest.pt"
    if not latest_ckpt.is_file():
        raise RuntimeError(f"Smoke run did not save latest checkpoint: {latest_ckpt}")
    if not metrics_file.is_file():
        raise RuntimeError(f"Smoke run did not write metrics.jsonl: {metrics_file}")
    metrics_text = metrics_file.read_text(encoding="utf-8")
    if '"event": "validation"' not in metrics_text:
        raise RuntimeError(f"Smoke run did not reach validation: {metrics_file}")


def _assert_run_outputs_inside_root(run_root: Path, cfg_path: Path) -> None:
    cfg = _load_yaml(cfg_path)
    train_cfg = cfg.get("training", {}) or {}
    for key in (
        "log_dir",
        "checkpoint_dir",
        "verbose_log_file",
        "metrics_jsonl_file",
        "run_metadata_file",
        "config_snapshot_file",
    ):
        value = train_cfg.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if run_root.resolve() not in path.parents and path != run_root.resolve():
            raise RuntimeError(
                f"Generated path for {key} escapes run root {run_root}: {path}"
            )


def launch_suite(
    *,
    project_root: Path,
    output_root: Path,
    smoke_timeout_s: int = 6 * 3600,
    full_start_timeout_s: int = 15 * 60,
    submit_baseline_full: bool = False,
) -> dict[str, Any]:
    manifest = _load_yaml(output_root / "manifest.yaml")
    smoke_jobs: dict[str, str] = {}
    full_jobs: dict[str, str] = {}
    full_run_names = set(SUITE_RUNS if submit_baseline_full else OPTION_RUNS)

    for run in manifest.get("runs", []):
        run_root = Path(run["root"])
        smoke_job_id = _submit_job(
            project_root=project_root,
            config_path=Path(run["smoke_config"]),
            runner_script=Path(run["smoke_runner"]),
            log_dir=run_root / "smoke" / "slurm_logs",
            job_name=f"fm6_{run['name']}_smoke",
        )
        smoke_jobs[run["name"]] = smoke_job_id

    for run in manifest.get("runs", []):
        run_root = Path(run["root"])
        _wait_for_state(
            smoke_jobs[run["name"]], {"COMPLETED"}, timeout_s=smoke_timeout_s
        )
        _assert_smoke_outputs(run_root)
        _assert_run_outputs_inside_root(run_root, Path(run["smoke_config"]))

    for run in manifest.get("runs", []):
        if run["name"] not in full_run_names:
            continue
        run_root = Path(run["root"])
        full_job_id = _submit_job(
            project_root=project_root,
            config_path=Path(run["full_config"]),
            runner_script=Path(run["full_runner"]),
            log_dir=run_root / "full" / "slurm_logs",
            job_name=f"fm6_{run['name']}_full",
        )
        full_jobs[run["name"]] = full_job_id

    started = {}
    for run in manifest.get("runs", []):
        if run["name"] not in full_run_names:
            continue
        state = _wait_for_state(
            full_jobs[run["name"]],
            {"RUNNING", "COMPLETED"},
            timeout_s=full_start_timeout_s,
        )
        started[run["name"]] = state

    launch_report = {
        "smoke_jobs": smoke_jobs,
        "full_jobs": full_jobs,
        "full_start_states": started,
        "submit_baseline_full": submit_baseline_full,
    }
    (output_root / "launch_report.json").write_text(
        json.dumps(launch_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return launch_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or launch the FocusMamba six-run suite."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    generate_parser.add_argument("--output-root", type=Path, default=Path("six_runs"))
    generate_parser.add_argument(
        "--full-template",
        type=Path,
        default=Path("configs/experiments/tartanair_focusmamba_v2_distill_50ep.yaml"),
    )
    generate_parser.add_argument(
        "--smoke-template",
        type=Path,
        default=Path("configs/experiments/tartanair_focusmamba_v2_smoke_distill.yaml"),
    )

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    launch_parser.add_argument("--output-root", type=Path, default=Path("six_runs"))
    launch_parser.add_argument("--smoke-timeout-s", type=int, default=6 * 3600)
    launch_parser.add_argument("--full-start-timeout-s", type=int, default=15 * 60)
    launch_parser.add_argument("--submit-baseline-full", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_suite(
            project_root=args.project_root.resolve(),
            output_root=args.output_root.resolve(),
            full_template=(args.project_root / args.full_template).resolve()
            if not args.full_template.is_absolute()
            else args.full_template.resolve(),
            smoke_template=(args.project_root / args.smoke_template).resolve()
            if not args.smoke_template.is_absolute()
            else args.smoke_template.resolve(),
        )
        return 0
    if args.command == "launch":
        launch_suite(
            project_root=args.project_root.resolve(),
            output_root=args.output_root.resolve(),
            smoke_timeout_s=args.smoke_timeout_s,
            full_start_timeout_s=args.full_start_timeout_s,
            submit_baseline_full=bool(args.submit_baseline_full),
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
