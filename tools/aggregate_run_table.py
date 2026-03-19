#!/usr/bin/env python3
"""Aggregate training runs into a ranked summary table.

Scans:
- runs/**/train_verbose.log
- logs/*.out / logs/*.err

Outputs:
- ranked markdown table to stdout
- optional CSV (via --csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml


VAL_RE = re.compile(
    r"Validation epoch\s+\d+\s+\|\s+loss=([0-9eE+\-.]+)\s+abs_rel=([0-9eE+\-.]+)\s+delta1=([0-9eE+\-.]+)\s+rmse=([0-9eE+\-.]+)"
)
PARAM_RE = re.compile(r"Parameters:\s*([0-9,]+)")
FPS_RE = re.compile(r"Latency:\s*([0-9eE+\-.]+)\s*FPS")
TB_RE = re.compile(r"TensorBoard logs ->\s*(runs/[^\s]+)")
JOBID_RE = re.compile(r"_([0-9]{6,})\.(out|err)$")
RESUME_SKIP_RE = re.compile(r"start_epoch=(\d+)\s+max_epochs=(\d+)")
ARGS_CFG_RE = re.compile(r"Args:\s+config=([^\s]+)")
TARGET_RE = re.compile(r"'training_target':\s*'([^']+)'")


def _load_target_mode(root: Path, config_path: str | None) -> Optional[str]:
    if not config_path:
        return None
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.is_file():
        return None
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return None
    target_mode = cfg.get("loss", {}).get("training_target")
    if target_mode is None:
        return None
    return str(target_mode).strip().lower()


def _load_run_tags(root: Path, config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.is_file():
        return {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    distill_cfg = cfg.get("distillation", {})
    recipe = str(distill_cfg.get("strategy", "none")).strip().lower()
    prefilter_enabled = bool(model_cfg.get("prefilter_enabled", False))
    prefilter = (
        str(model_cfg.get("prefilter_type", "none")).strip().lower()
        if prefilter_enabled
        else "none"
    )
    gate_enabled = bool(model_cfg.get("state_gate_enabled", False))
    gate_mask = model_cfg.get("state_gate_stage_mask")
    gate = "off"
    if gate_enabled:
        gate = "all"
        if gate_mask is not None:
            mask = [bool(v) for v in gate_mask]
            gate = "last" if mask == [False, False, False, True] else "masked"

    num_frames = int(model_cfg.get("num_frames", 0) or 0)
    reset_interval = int(model_cfg.get("stream_reset_interval", 0) or 0)
    cache_len = int(model_cfg.get("stream_max_cache_len", 0) or 0)
    heuristic = "baseline"
    if reset_interval > 0:
        heuristic = f"reset{reset_interval}"
    elif cache_len > 0 and (num_frames == 0 or cache_len < num_frames):
        heuristic = f"k{cache_len}"

    temporal_adapter = "off"
    if bool(model_cfg.get("pre_temporal_stage_adapter_enabled", False)):
        raw_stages = model_cfg.get(
            "pre_temporal_stage_adapter_stages", ["layer3", "layer4"]
        )
        if raw_stages is None:
            raw_stages = ["layer3", "layer4"]
        normalized_stages = sorted(
            {str(stage).strip().lower().replace("_", "") for stage in raw_stages}
        )
        temporal_adapter = "+".join(normalized_stages) if normalized_stages else "on"

    train_budget = data_cfg.get(
        "max_train_trajectories", data_cfg.get("max_trajectories")
    )
    val_budget = data_cfg.get("max_val_trajectories", data_cfg.get("max_trajectories"))
    budget_scope = "full" if train_budget is None and val_budget is None else "subset"

    return {
        "recipe_family": recipe,
        "prefilter": prefilter,
        "gate": gate,
        "temporal_adapter": temporal_adapter,
        "reset_interval": reset_interval,
        "cache_len": cache_len,
        "heuristic": heuristic,
        "train_budget": train_budget,
        "val_budget": val_budget,
        "budget_scope": budget_scope,
    }


def _parse_ts_prefix(line: str) -> Optional[datetime]:
    # Example: 2026-03-05 20:25:40 | INFO | ...
    try:
        return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _parse_iso_ts(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


@dataclass
class RunRecord:
    run_dir: str
    params: Optional[int] = None
    fps: Optional[float] = None
    best_loss: Optional[float] = None
    best_abs_rel: Optional[float] = None
    best_delta1: Optional[float] = None
    best_rmse: Optional[float] = None
    best_si_log: Optional[float] = None
    best_fdv: Optional[float] = None
    completed: bool = False
    failed: bool = False
    skip_noop_resume: bool = False
    job_ids: set[str] = field(default_factory=set)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    sources: List[str] = field(default_factory=list)
    fail_markers: List[str] = field(default_factory=list)
    config_path: Optional[str] = None
    target_mode: Optional[str] = None
    recipe_family: Optional[str] = None
    prefilter: Optional[str] = None
    gate: Optional[str] = None
    temporal_adapter: Optional[str] = None
    reset_interval: Optional[int] = None
    cache_len: Optional[int] = None
    heuristic: Optional[str] = None
    train_budget: Optional[int] = None
    val_budget: Optional[int] = None
    budget_scope: Optional[str] = None

    @property
    def runtime_min(self) -> Optional[float]:
        if self.first_ts and self.last_ts and self.last_ts >= self.first_ts:
            return (self.last_ts - self.first_ts).total_seconds() / 60.0
        return None

    @property
    def status(self) -> str:
        if self.failed:
            return "failed"
        if self.skip_noop_resume:
            return "noop_resume"
        if self.completed:
            return "completed"
        return "incomplete"


def _ensure_record(
    records: Dict[str, RunRecord], run_dir: str, source: str
) -> RunRecord:
    rec = records.get(run_dir)
    if rec is None:
        rec = RunRecord(run_dir=run_dir)
        records[run_dir] = rec
    if source not in rec.sources:
        rec.sources.append(source)
    return rec


def parse_verbose_logs(root: Path, records: Dict[str, RunRecord]) -> None:
    for p in sorted(root.glob("runs/**/train_verbose.log")):
        run_dir = str(p.parent).replace(str(root) + "/", "")
        rec = _ensure_record(records, run_dir, str(p))

        for line in p.read_text(errors="ignore").splitlines():
            if rec.config_path is None:
                m_cfg = ARGS_CFG_RE.search(line)
                if m_cfg:
                    rec.config_path = m_cfg.group(1)
                    rec.target_mode = rec.target_mode or _load_target_mode(
                        root, rec.config_path
                    )
                    for key, value in _load_run_tags(root, rec.config_path).items():
                        setattr(rec, key, value)
            if rec.target_mode is None and "training_target" in line:
                m_target = TARGET_RE.search(line)
                if m_target:
                    rec.target_mode = m_target.group(1).strip().lower()

            ts = _parse_ts_prefix(line)
            if ts is not None:
                if rec.first_ts is None or ts < rec.first_ts:
                    rec.first_ts = ts
                if rec.last_ts is None or ts > rec.last_ts:
                    rec.last_ts = ts

            if "Student:" in line and "Parameters:" in line:
                m = PARAM_RE.search(line)
                if m:
                    rec.params = int(m.group(1).replace(",", ""))

            if "Latency:" in line and "FPS" in line:
                m = FPS_RE.search(line)
                if m:
                    rec.fps = float(m.group(1))

            if "Validation epoch" in line:
                m = VAL_RE.search(line)
                if m:
                    loss = float(m.group(1))
                    abs_rel = float(m.group(2))
                    delta1 = float(m.group(3))
                    rmse = float(m.group(4))
                    if rec.best_abs_rel is None or abs_rel < rec.best_abs_rel:
                        rec.best_abs_rel = abs_rel
                        rec.best_delta1 = delta1
                        rec.best_rmse = rmse
                        rec.best_loss = loss

            if "Training complete." in line:
                rec.completed = True

            if "Traceback" in line or "RuntimeError:" in line:
                rec.failed = True
                rec.fail_markers.append("verbose_traceback")


def parse_run_metadata(root: Path, records: Dict[str, RunRecord]) -> None:
    for p in sorted(root.glob("runs/**/run_metadata.json")):
        run_dir = str(p.parent).replace(str(root) + "/", "")
        rec = _ensure_record(records, run_dir, str(p))
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        rec.config_path = rec.config_path or payload.get("config_path")
        target_mode = payload.get("target_mode")
        if target_mode:
            rec.target_mode = rec.target_mode or str(target_mode).strip().lower()
        for key, value in _load_run_tags(root, rec.config_path).items():
            setattr(rec, key, value)
        ts = _parse_iso_ts(payload.get("created_at"))
        if ts is not None:
            if rec.first_ts is None or ts < rec.first_ts:
                rec.first_ts = ts


def parse_metrics_jsonl(root: Path, records: Dict[str, RunRecord]) -> None:
    for p in sorted(root.glob("runs/**/metrics.jsonl")):
        run_dir = str(p.parent).replace(str(root) + "/", "")
        rec = _ensure_record(records, run_dir, str(p))

        for line in p.read_text(errors="ignore").splitlines():
            try:
                payload = json.loads(line)
            except Exception:
                continue

            ts = _parse_iso_ts(payload.get("ts"))
            if ts is not None:
                if rec.first_ts is None or ts < rec.first_ts:
                    rec.first_ts = ts
                if rec.last_ts is None or ts > rec.last_ts:
                    rec.last_ts = ts

            event = payload.get("event")
            if event == "run_start":
                trainable_params = payload.get("trainable_params")
                if trainable_params is not None:
                    rec.params = int(trainable_params)
                target_mode = payload.get("target_mode")
                if target_mode:
                    rec.target_mode = (
                        rec.target_mode or str(target_mode).strip().lower()
                    )
            elif event == "validation":
                metrics = payload.get("metrics") or {}
                abs_rel = metrics.get("abs_rel")
                if abs_rel is None:
                    continue
                loss = metrics.get("loss")
                delta1 = metrics.get("delta1")
                rmse = metrics.get("rmse")
                si_log = metrics.get("si_log")
                fdv = metrics.get("fdv")
                if rec.best_abs_rel is None or float(abs_rel) < rec.best_abs_rel:
                    rec.best_abs_rel = float(abs_rel)
                    rec.best_delta1 = float(delta1) if delta1 is not None else None
                    rec.best_rmse = float(rmse) if rmse is not None else None
                    rec.best_si_log = float(si_log) if si_log is not None else None
                    rec.best_fdv = float(fdv) if fdv is not None else None
                    rec.best_loss = float(loss) if loss is not None else None
            elif event == "latency":
                metrics = payload.get("metrics") or {}
                fps = metrics.get("fps")
                if fps is not None:
                    rec.fps = float(fps)
            elif event == "run_complete":
                rec.completed = True


def parse_slurm_logs(root: Path, records: Dict[str, RunRecord]) -> None:
    for p in sorted(root.glob("logs/*.out")):
        text = p.read_text(errors="ignore")
        run_dir = None

        m_tb = TB_RE.search(text)
        if m_tb:
            run_dir = m_tb.group(1)

        if run_dir is None:
            continue

        rec = _ensure_record(records, run_dir, str(p))

        m_job = JOBID_RE.search(p.name)
        if m_job:
            rec.job_ids.add(m_job.group(1))

        for line in text.splitlines():
            if rec.config_path is None:
                m_cfg = ARGS_CFG_RE.search(line)
                if m_cfg:
                    rec.config_path = m_cfg.group(1)
                    rec.target_mode = rec.target_mode or _load_target_mode(
                        root, rec.config_path
                    )
            if rec.target_mode is None and "training_target" in line:
                m_target = TARGET_RE.search(line)
                if m_target:
                    rec.target_mode = m_target.group(1).strip().lower()

            ts = _parse_ts_prefix(line)
            if ts is not None:
                if rec.first_ts is None or ts < rec.first_ts:
                    rec.first_ts = ts
                if rec.last_ts is None or ts > rec.last_ts:
                    rec.last_ts = ts

            if "Student:" in line and "Parameters:" in line:
                m = PARAM_RE.search(line)
                if m:
                    rec.params = int(m.group(1).replace(",", ""))

            if "Latency:" in line and "FPS" in line:
                m = FPS_RE.search(line)
                if m:
                    rec.fps = float(m.group(1))

            if "Validation epoch" in line:
                m = VAL_RE.search(line)
                if m:
                    loss = float(m.group(1))
                    abs_rel = float(m.group(2))
                    delta1 = float(m.group(3))
                    rmse = float(m.group(4))
                    if rec.best_abs_rel is None or abs_rel < rec.best_abs_rel:
                        rec.best_abs_rel = abs_rel
                        rec.best_delta1 = delta1
                        rec.best_rmse = rmse
                        rec.best_loss = loss

            if "Resumed from epoch" in line:
                pass
            if "Training loop:" in line:
                m_skip = RESUME_SKIP_RE.search(line)
                if m_skip:
                    start_epoch = int(m_skip.group(1))
                    max_epochs = int(m_skip.group(2))
                    if start_epoch >= max_epochs:
                        rec.skip_noop_resume = True

            if "Training complete." in line:
                rec.completed = True

            if "Traceback" in line or "RuntimeError:" in line:
                rec.failed = True
                rec.fail_markers.append("slurm_out_traceback")

    for p in sorted(root.glob("logs/*.err")):
        text = p.read_text(errors="ignore")
        m_tb = TB_RE.search(text)
        run_dir = m_tb.group(1) if m_tb else None
        if run_dir is None:
            # No mapping; skip.
            continue
        rec = _ensure_record(records, run_dir, str(p))
        m_job = JOBID_RE.search(p.name)
        if m_job:
            rec.job_ids.add(m_job.group(1))
        if "Traceback" in text or "RuntimeError:" in text:
            rec.failed = True
            rec.fail_markers.append("slurm_err_traceback")


def _rank_group(records: List[RunRecord]) -> List[RunRecord]:
    ranked = [
        r
        for r in records
        if r.best_abs_rel is not None and r.run_dir not in {"runs", "."}
    ]
    ranked.sort(key=lambda r: (r.best_abs_rel, -(r.best_delta1 or 0.0), r.run_dir))
    return ranked


def print_ranked(
    records: Dict[str, RunRecord],
    top_k: int = 200,
    mode: str = "all",
) -> List[tuple[str, int, RunRecord]]:
    grouped: Dict[str, List[RunRecord]] = defaultdict(list)
    for rec in records.values():
        grouped[(rec.target_mode or "unknown")].append(rec)

    if mode == "all":
        mode_order = [m for m in ("metric", "relative", "unknown") if grouped.get(m)]
    else:
        mode_order = [mode]

    ranked_rows: List[tuple[str, int, RunRecord]] = []
    first_section = True
    for target_mode in mode_order:
        ranked = _rank_group(grouped.get(target_mode, []))
        if not ranked:
            continue
        if not first_section:
            print()
        first_section = False
        title = f"{target_mode.title()} runs"
        print(title)
        print(
            "| rank | mode | run | recipe | prefilter | temporal | gate | budget | heuristic | status | abs_rel | delta1 | rmse | si_log | fdv | fps | params | runtime_min | jobs |"
        )
        print(
            "|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        )
        for i, r in enumerate(ranked[:top_k], start=1):
            params = str(r.params) if r.params is not None else "-"
            fps = f"{r.fps:.1f}" if r.fps is not None else "-"
            runtime = f"{r.runtime_min:.1f}" if r.runtime_min is not None else "-"
            jobs = ",".join(sorted(r.job_ids)) if r.job_ids else "-"
            train_budget = "full" if r.train_budget is None else str(r.train_budget)
            val_budget = "full" if r.val_budget is None else str(r.val_budget)
            budget = f"tr={train_budget}/val={val_budget}"
            print(
                f"| {i} | {target_mode} | {r.run_dir} | {r.recipe_family or '-'} | "
                f"{r.prefilter or '-'} | {r.temporal_adapter or '-'} | {r.gate or '-'} | "
                f"{budget} | {r.heuristic or '-'} | {r.status} | "
                f"{r.best_abs_rel:.5f} | {r.best_delta1 or 0.0:.4f} | {r.best_rmse or 0.0:.4f} | "
                f"{r.best_si_log or 0.0:.4f} | {r.best_fdv or 0.0:.4f} | "
                f"{fps} | {params} | {runtime} | {jobs} |"
            )
            ranked_rows.append((target_mode, i, r))
    return ranked_rows


def print_fail_patterns(records: Dict[str, RunRecord]) -> None:
    buckets = defaultdict(list)
    for r in records.values():
        if r.failed:
            buckets["failed"].append(r)
        elif r.skip_noop_resume:
            buckets["noop_resume"].append(r)
        elif not r.completed and r.best_abs_rel is not None:
            buckets["incomplete_with_val"].append(r)
        elif not r.completed:
            buckets["incomplete_no_val"].append(r)

    print("\nFailure/health summary:")
    for k in ("failed", "noop_resume", "incomplete_with_val", "incomplete_no_val"):
        print(f"- {k}: {len(buckets.get(k, []))}")

    if buckets.get("noop_resume"):
        print("  noop_resume runs:")
        for r in sorted(buckets["noop_resume"], key=lambda x: x.run_dir):
            print(
                f"  - {r.run_dir} jobs={','.join(sorted(r.job_ids)) if r.job_ids else '-'}"
            )

    if buckets.get("failed"):
        print("  failed runs:")
        for r in sorted(buckets["failed"], key=lambda x: x.run_dir):
            marks = ",".join(sorted(set(r.fail_markers))) if r.fail_markers else "-"
            print(f"  - {r.run_dir} markers={marks}")


def write_csv(path: Path, rows: List[tuple[str, int, RunRecord]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "target_mode",
                "run",
                "recipe_family",
                "prefilter",
                "temporal_adapter",
                "gate",
                "train_budget",
                "val_budget",
                "budget_scope",
                "heuristic",
                "status",
                "abs_rel",
                "delta1",
                "rmse",
                "si_log",
                "fdv",
                "fps",
                "params",
                "runtime_min",
                "job_ids",
            ]
        )
        for target_mode, rank, r in rows:
            w.writerow(
                [
                    rank,
                    target_mode,
                    r.run_dir,
                    r.recipe_family or "",
                    r.prefilter or "",
                    r.temporal_adapter or "",
                    r.gate or "",
                    "" if r.train_budget is None else r.train_budget,
                    "" if r.val_budget is None else r.val_budget,
                    r.budget_scope or "",
                    r.heuristic or "",
                    r.status,
                    f"{r.best_abs_rel:.6f}" if r.best_abs_rel is not None else "",
                    f"{r.best_delta1:.6f}" if r.best_delta1 is not None else "",
                    f"{r.best_rmse:.6f}" if r.best_rmse is not None else "",
                    f"{r.best_si_log:.6f}" if r.best_si_log is not None else "",
                    f"{r.best_fdv:.6f}" if r.best_fdv is not None else "",
                    f"{r.fps:.3f}" if r.fps is not None else "",
                    r.params if r.params is not None else "",
                    f"{r.runtime_min:.3f}" if r.runtime_min is not None else "",
                    ",".join(sorted(r.job_ids)),
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate FocusMamba run logs into a ranked table."
    )
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument(
        "--mode",
        choices=("all", "metric", "relative", "unknown"),
        default="all",
        help="Filter rankings by loss.training_target mode.",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    records: Dict[str, RunRecord] = {}
    parse_run_metadata(root, records)
    parse_metrics_jsonl(root, records)
    parse_verbose_logs(root, records)
    parse_slurm_logs(root, records)

    ranked = print_ranked(records, top_k=args.top_k, mode=args.mode)
    print_fail_patterns(records)

    if args.csv is not None:
        csv_path = args.csv if args.csv.is_absolute() else (root / args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, ranked)
        print(f"\nCSV written: {csv_path}")


if __name__ == "__main__":
    main()
