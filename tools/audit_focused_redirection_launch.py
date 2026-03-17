#!/usr/bin/env python3
"""Audit a focused-redirection launch set against running and recent runs.

Builds a method-level overlap report so new submissions do not duplicate
experiments that are already running or have already produced usable results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXISTING_ROOTS = [
    "runs/acmmm_vda_streaming_metric_20260307_half_frames",
    "runs/deadline_rescue_sweep_20260309",
    "runs/teacher_loss_sweep_20260308",
    "runs/focused_redirection_20260310",
    "runs/focused_redirection_paperlock_20260310",
]

JOBID_RE = re.compile(r"_([0-9]{6,})\.(?:out|err)$")
CONFIG_RE = re.compile(r"config=([^\s]+)")
DEFAULT_STATS_ALIGN_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STATS_ALIGN_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MethodSignature:
    recipe: str
    prefilter: str
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]
    temporal_adapter: str
    gate: str
    gate_mask: tuple[bool, ...]
    reset_interval: int
    cache_len: int
    freeze_backbone: bool
    unfreeze_prefixes: tuple[str, ...]
    curriculum: str
    temporal_weight: float
    temporal_threshold: float
    clean_consistency_weight: float
    clean_consistency_mode: str
    train_budget: int | None
    val_budget: int | None
    max_epochs: int


@dataclass(frozen=True)
class FamilySignature:
    prefilter: str
    gate: str
    reset_interval: int
    cache_len: int
    curriculum: str
    temporal_weight: float
    temporal_threshold: float
    clean_consistency_weight: float
    clean_consistency_mode: str


@dataclass
class Candidate:
    job_name: str
    config_path: str
    name: str
    method_signature: MethodSignature
    family_signature: FamilySignature


@dataclass
class ExistingRun:
    suite: str
    run: str
    config_path: str
    method_signature: MethodSignature
    family_signature: FamilySignature
    status: str
    best_abs_rel: float | None
    best_epoch: int | None
    latest_event: str | None
    latest_epoch: int | None
    job_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--candidate-suite",
        default="configs/experiments/generated/focused_redirection_paperlock_20260310",
        help="Config directory containing jobs.tsv and generated YAML configs.",
    )
    parser.add_argument(
        "--existing-root",
        action="append",
        default=[],
        help="Run roots to inspect for duplicates. May be passed multiple times.",
    )
    parser.add_argument(
        "--audit-tsv",
        default=None,
        help="Output audit TSV path. Defaults to <candidate-suite>/launch_audit.tsv.",
    )
    parser.add_argument(
        "--launch-tsv",
        default=None,
        help="Output filtered jobs TSV path. Defaults to <candidate-suite>/launch_jobs.tsv.",
    )
    return parser.parse_args()


def _round_tuple(values: Any) -> tuple[float, ...]:
    if values is None:
        return ()
    return tuple(round(float(v), 6) for v in values)


def _normalize_curriculum(value: Any) -> str:
    if not value:
        return "off"
    if value is True:
        return "on"
    if value is False:
        return "off"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _gate_label(model_cfg: dict[str, Any]) -> tuple[str, tuple[bool, ...]]:
    if not bool(model_cfg.get("state_gate_enabled", False)):
        return "off", ()
    mask = model_cfg.get("state_gate_stage_mask")
    if mask is None:
        return "all", ()
    mask_tuple = tuple(bool(v) for v in mask)
    if mask_tuple == (False, False, False, True):
        return "last", mask_tuple
    return "masked", mask_tuple


def load_config(config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def build_method_signature(cfg: dict[str, Any]) -> MethodSignature:
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    distill_cfg = cfg.get("distillation", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    degradation_cfg = cfg.get("degradation", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    aux_cfg = loss_cfg.get("auxiliary", {}) or {}

    prefilter = "none"
    target_mean: tuple[float, ...] = ()
    target_std: tuple[float, ...] = ()
    if bool(model_cfg.get("prefilter_enabled", False)):
        prefilter = str(model_cfg.get("prefilter_type", "none")).strip().lower()
        if prefilter == "stats_align":
            target_mean = _round_tuple(model_cfg.get("prefilter_target_mean")) or DEFAULT_STATS_ALIGN_MEAN
            target_std = _round_tuple(model_cfg.get("prefilter_target_std")) or DEFAULT_STATS_ALIGN_STD

    temporal_adapter = "off"
    if bool(model_cfg.get("pre_temporal_stage_adapter_enabled", False)):
        raw_stages = model_cfg.get("pre_temporal_stage_adapter_stages", ["layer3", "layer4"]) or ["layer3", "layer4"]
        normalized_stages = sorted({str(stage).strip().lower().replace("_", "") for stage in raw_stages})
        temporal_adapter = "+".join(normalized_stages) if normalized_stages else "on"

    gate, gate_mask = _gate_label(model_cfg)

    return MethodSignature(
        recipe=str(distill_cfg.get("strategy", "none")).strip().lower(),
        prefilter=prefilter,
        target_mean=target_mean,
        target_std=target_std,
        temporal_adapter=temporal_adapter,
        gate=gate,
        gate_mask=gate_mask,
        reset_interval=int(model_cfg.get("stream_reset_interval", 0) or 0),
        cache_len=int(model_cfg.get("stream_max_cache_len", 0) or 0),
        freeze_backbone=bool(training_cfg.get("freeze_backbone", False)),
        unfreeze_prefixes=tuple(str(v) for v in (training_cfg.get("unfreeze_prefixes", []) or [])),
        curriculum=_normalize_curriculum(degradation_cfg.get("curriculum")),
        temporal_weight=round(float(loss_cfg.get("temporal_weight", 0.0) or 0.0), 6),
        temporal_threshold=round(float(loss_cfg.get("temporal_threshold", 0.0) or 0.0), 6),
        clean_consistency_weight=round(
            float(aux_cfg.get("clean_depth_consistency_weight", 0.0) or 0.0), 6
        ),
        clean_consistency_mode=str(aux_cfg.get("clean_depth_consistency_mode", "") or "").strip().lower(),
        train_budget=data_cfg.get("max_train_trajectories", data_cfg.get("max_trajectories")),
        val_budget=data_cfg.get("max_val_trajectories", data_cfg.get("max_trajectories")),
        max_epochs=int(training_cfg.get("max_epochs", 0) or 0),
    )


def to_family_signature(method: MethodSignature) -> FamilySignature:
    return FamilySignature(
        prefilter=method.prefilter,
        gate=method.gate,
        reset_interval=method.reset_interval,
        cache_len=method.cache_len,
        curriculum=method.curriculum,
        temporal_weight=method.temporal_weight,
        temporal_threshold=method.temporal_threshold,
        clean_consistency_weight=method.clean_consistency_weight,
        clean_consistency_mode=method.clean_consistency_mode,
    )


def heuristic_label(method: MethodSignature) -> str:
    if method.reset_interval > 0:
        return f"reset-{method.reset_interval}"
    if method.cache_len > 0 and method.cache_len < 16:
        return f"K={method.cache_len}"
    if method.gate != "off":
        return "gate"
    return "baseline"


def load_candidates(project_root: Path, suite_dir: Path) -> list[Candidate]:
    jobs_path = suite_dir / "jobs.tsv"
    candidates: list[Candidate] = []
    with jobs_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            config_rel = row["config_path"].strip()
            config_path = project_root / config_rel
            cfg = load_config(config_path)
            method = build_method_signature(cfg)
            candidates.append(
                Candidate(
                    job_name=row["job_name"].strip(),
                    config_path=config_rel,
                    name=Path(config_rel).stem,
                    method_signature=method,
                    family_signature=to_family_signature(method),
                )
            )
    return candidates


def _run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return proc.stdout


def get_running_job_ids(project_root: Path) -> set[str]:
    try:
        out = _run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"], project_root)
    except Exception:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def infer_config_path(project_root: Path, run_dir: Path) -> Path | None:
    verbose_path = run_dir / "train_verbose.log"
    if verbose_path.is_file():
        for line in verbose_path.read_text(errors="ignore").splitlines():
            match = CONFIG_RE.search(line)
            if match:
                config_path = Path(match.group(1).strip())
                return config_path if config_path.is_absolute() else project_root / config_path
    generated_guess = project_root / "configs" / "experiments" / "generated" / run_dir.parent.name / f"{run_dir.name}.yaml"
    if generated_guess.is_file():
        return generated_guess
    return None


def read_metrics(metrics_path: Path) -> tuple[float | None, int | None, str | None, int | None]:
    best_abs_rel = None
    best_epoch = None
    latest_event = None
    latest_epoch = None
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        latest_event = payload.get("event")
        latest_epoch = payload.get("epoch")
        if payload.get("event") != "validation":
            continue
        value = payload.get("metrics", {}).get("abs_rel")
        if value is None:
            continue
        if best_abs_rel is None or float(value) < best_abs_rel:
            best_abs_rel = float(value)
            best_epoch = payload.get("epoch")
    return best_abs_rel, best_epoch, latest_event, latest_epoch


def read_job_ids(run_dir: Path) -> tuple[str, ...]:
    job_ids: set[str] = set()
    for path in (run_dir / "logs").glob("*"):
        match = JOBID_RE.search(path.name)
        if match:
            job_ids.add(match.group(1))
    return tuple(sorted(job_ids))


def infer_status(run_dir: Path, latest_event: str | None, running_job_ids: set[str]) -> str:
    if latest_event == "run_complete":
        return "completed"
    job_ids = read_job_ids(run_dir)
    if any(job_id in running_job_ids for job_id in job_ids):
        return "running"
    if latest_event == "validation":
        return "stopped_with_val"
    if latest_event == "train_epoch":
        return "stopped_train_only"
    return "unknown"


def load_existing_runs(project_root: Path, roots: list[Path], running_job_ids: set[str]) -> list[ExistingRun]:
    records: list[ExistingRun] = []
    for root in roots:
        if not root.exists():
            continue
        for metrics_path in sorted(root.glob("*/metrics.jsonl")):
            run_dir = metrics_path.parent
            config_path = infer_config_path(project_root, run_dir)
            if config_path is None or not config_path.is_file():
                continue
            cfg = load_config(config_path)
            method = build_method_signature(cfg)
            best_abs_rel, best_epoch, latest_event, latest_epoch = read_metrics(metrics_path)
            status = infer_status(run_dir, latest_event, running_job_ids)
            records.append(
                ExistingRun(
                    suite=root.name,
                    run=run_dir.name,
                    config_path=str(config_path.relative_to(project_root)),
                    method_signature=method,
                    family_signature=to_family_signature(method),
                    status=status,
                    best_abs_rel=best_abs_rel,
                    best_epoch=best_epoch,
                    latest_event=latest_event,
                    latest_epoch=latest_epoch,
                    job_ids=read_job_ids(run_dir),
                )
            )
    return records


def select_best_match(matches: list[ExistingRun]) -> ExistingRun | None:
    if not matches:
        return None
    status_rank = {
        "running": 0,
        "completed": 1,
        "stopped_with_val": 2,
        "stopped_train_only": 3,
        "unknown": 4,
    }
    return sorted(
        matches,
        key=lambda rec: (
            status_rank.get(rec.status, 9),
            rec.best_abs_rel if rec.best_abs_rel is not None else float("inf"),
            rec.run,
        ),
    )[0]


def decide(candidate: Candidate, existing_runs: list[ExistingRun]) -> dict[str, str]:
    exact_matches = [rec for rec in existing_runs if rec.method_signature == candidate.method_signature]
    related_matches = [rec for rec in existing_runs if rec.family_signature == candidate.family_signature]
    exact_running = [rec for rec in exact_matches if rec.status == "running"]
    exact_completed = [rec for rec in exact_matches if rec.status in {"completed", "stopped_with_val"}]

    decision = "launch_now"
    reason = "no exact method match in running or recent runs"
    match = None
    if exact_running:
        decision = "skip_running_exact"
        reason = "exact method already running"
        match = select_best_match(exact_running)
    elif exact_completed:
        decision = "skip_recent_exact"
        reason = "exact method already has recent results"
        match = select_best_match(exact_completed)
    elif exact_matches:
        decision = "review_exact_unknown"
        reason = "exact method exists but status is unclear"
        match = select_best_match(exact_matches)

    related = select_best_match(related_matches)
    return {
        "decision": decision,
        "reason": reason,
        "match_suite": match.suite if match else "",
        "match_run": match.run if match else "",
        "match_status": match.status if match else "",
        "match_abs_rel": "" if match is None or match.best_abs_rel is None else f"{match.best_abs_rel:.6f}",
        "related_suite": related.suite if related else "",
        "related_run": related.run if related else "",
        "related_status": related.status if related else "",
        "related_abs_rel": "" if related is None or related.best_abs_rel is None else f"{related.best_abs_rel:.6f}",
    }


def write_audit(
    audit_path: Path,
    launch_path: Path,
    candidates: list[Candidate],
    existing_runs: list[ExistingRun],
) -> tuple[int, int]:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    launch_path.parent.mkdir(parents=True, exist_ok=True)

    launch_rows: list[tuple[str, str]] = []
    submitted = 0

    with audit_path.open("w", newline="") as audit_file:
        writer = csv.writer(audit_file, delimiter="\t")
        writer.writerow(
            [
                "decision",
                "job_name",
                "candidate",
                "config_path",
                "recipe",
                "prefilter",
                "gate",
                "heuristic",
                "match_suite",
                "match_run",
                "match_status",
                "match_abs_rel",
                "related_suite",
                "related_run",
                "related_status",
                "related_abs_rel",
                "reason",
            ]
        )
        for candidate in candidates:
            result = decide(candidate, existing_runs)
            writer.writerow(
                [
                    result["decision"],
                    candidate.job_name,
                    candidate.name,
                    candidate.config_path,
                    candidate.method_signature.recipe,
                    candidate.method_signature.prefilter,
                    candidate.method_signature.gate,
                    heuristic_label(candidate.method_signature),
                    result["match_suite"],
                    result["match_run"],
                    result["match_status"],
                    result["match_abs_rel"],
                    result["related_suite"],
                    result["related_run"],
                    result["related_status"],
                    result["related_abs_rel"],
                    result["reason"],
                ]
            )
            if result["decision"] == "launch_now":
                launch_rows.append((candidate.job_name, candidate.config_path))
                submitted += 1

    with launch_path.open("w", newline="") as launch_file:
        writer = csv.writer(launch_file, delimiter="\t")
        writer.writerow(["job_name", "config_path"])
        writer.writerows(launch_rows)

    return submitted, len(candidates)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    suite_dir = Path(args.candidate_suite)
    if not suite_dir.is_absolute():
        suite_dir = project_root / suite_dir
    existing_roots = [Path(p) if Path(p).is_absolute() else project_root / p for p in (args.existing_root or [])]
    if not existing_roots:
        existing_roots = [project_root / p for p in DEFAULT_EXISTING_ROOTS]

    audit_path = Path(args.audit_tsv) if args.audit_tsv else suite_dir / "launch_audit.tsv"
    launch_path = Path(args.launch_tsv) if args.launch_tsv else suite_dir / "launch_jobs.tsv"
    if not audit_path.is_absolute():
        audit_path = project_root / audit_path
    if not launch_path.is_absolute():
        launch_path = project_root / launch_path

    candidates = load_candidates(project_root, suite_dir)
    running_job_ids = get_running_job_ids(project_root)
    existing_runs = load_existing_runs(project_root, existing_roots, running_job_ids)
    launch_count, total = write_audit(audit_path, launch_path, candidates, existing_runs)

    print(f"Audited {total} candidate jobs against {len(existing_runs)} recent runs.")
    print(f"Launch-now jobs: {launch_count}")
    print(f"Audit TSV: {audit_path.relative_to(project_root)}")
    print(f"Launch TSV: {launch_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
