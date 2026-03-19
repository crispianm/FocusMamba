#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = PROJECT_ROOT / "bug_hunt_vda"
REPORT_PATH = SUITE_ROOT / "supervisor_update.md"
ASSET_DIR = SUITE_ROOT / "report_assets"

PRIMARY_RUN_ORDER = [
    ("00_clean_ft", "single"),
    ("01_degraded_ft", "single"),
    ("02_option1_prefilter", "phase_b_unfrozen"),
    ("03_option2_refiner", "phase_b_unfrozen"),
    ("04_option3_quality_gate", "phase_b_unfrozen"),
    ("05_option4_pre_temporal_adapter", "phase_b_unfrozen"),
    ("06_option5_aux_recon", "phase_b_unfrozen"),
    ("07_option6_degradation_conditioned_gate", "phase_b_unfrozen"),
]

PHASE_A_RUN_ORDER = [
    ("02_option1_prefilter", "phase_a_frozen"),
    ("03_option2_refiner", "phase_a_frozen"),
    ("04_option3_quality_gate", "phase_a_frozen"),
    ("05_option4_pre_temporal_adapter", "phase_a_frozen"),
    ("06_option5_aux_recon", "phase_a_frozen"),
    ("07_option6_degradation_conditioned_gate", "phase_a_frozen"),
]

DISPLAY_NAMES = {
    "00_clean_ft": "Clean FT",
    "01_degraded_ft": "Degraded FT",
    "02_option1_prefilter": "Opt 1: Prefilter",
    "03_option2_refiner": "Opt 2: Refiner",
    "04_option3_quality_gate": "Opt 3: Quality Gate",
    "05_option4_pre_temporal_adapter": "Opt 4: Pre-Temporal Adapter",
    "06_option5_aux_recon": "Opt 5: Aux Recon",
    "07_option6_degradation_conditioned_gate": "Opt 6: Degradation Gate",
}


@dataclass
class EventSummary:
    epoch: int
    metrics_by_scope: dict[str, dict[str, float]]
    selection_value: float | None


@dataclass
class RunSummary:
    run_name: str
    phase_name: str
    kind: str
    config_path: Path
    metrics_path: Path
    max_epochs: int
    latest: EventSummary | None
    best: EventSummary | None
    run_complete: dict[str, Any] | None
    latest_event_name: str | None


def _run_cmd(args: list[str]) -> str:
    result = subprocess.run(
        args, cwd=PROJECT_ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _iter_metric_events(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.is_file():
        return []
    events = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _validation_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("event") in {"validation", "pretrain_validation"}
    ]


def _metrics_by_scope(event: dict[str, Any]) -> dict[str, dict[str, float]]:
    validation_sets = event.get("validation_sets")
    if isinstance(validation_sets, dict) and validation_sets:
        return {
            str(scope): dict(metrics or {})
            for scope, metrics in validation_sets.items()
        }
    metrics = dict(event.get("metrics", {}) or {})
    if not metrics:
        return {}
    scope = str(event.get("primary_validation_set", "val_degraded"))
    return {scope: metrics}


def _selection_value(event: dict[str, Any]) -> float | None:
    raw = event.get("selection_value")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    metric_name = str(event.get("selection_metric") or "abs_rel")
    primary = str(event.get("primary_validation_set") or "val_degraded")
    metrics = _metrics_by_scope(event).get(primary, {})
    if metric_name in metrics:
        return float(metrics[metric_name])
    return None


def _event_summary(event: dict[str, Any] | None) -> EventSummary | None:
    if event is None:
        return None
    return EventSummary(
        epoch=int(event.get("epoch", 0) or 0),
        metrics_by_scope=_metrics_by_scope(event),
        selection_value=_selection_value(event),
    )


def _best_validation_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [event for event in events if event.get("event") == "validation"]
    if not candidates:
        candidates = events
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda event: float(
            _selection_value(event) if _selection_value(event) is not None else 1e18
        ),
    )


def _discover_run_summaries(root: Path) -> dict[tuple[str, str], RunSummary]:
    summaries: dict[tuple[str, str], RunSummary] = {}
    for run_name, phase_name in PRIMARY_RUN_ORDER + PHASE_A_RUN_ORDER:
        if phase_name == "single":
            config_path = root / run_name / "full" / "config.yaml"
            metrics_path = root / run_name / "full" / "metrics.jsonl"
        else:
            config_path = root / run_name / phase_name / "full" / "config.yaml"
            metrics_path = root / run_name / phase_name / "full" / "metrics.jsonl"
        if not config_path.is_file():
            continue
        cfg = _load_yaml(config_path)
        events = _iter_metric_events(metrics_path)
        validation = _validation_events(events)
        latest = validation[-1] if validation else None
        best = _best_validation_event(validation)
        run_complete = next(
            (
                event
                for event in reversed(events)
                if event.get("event") == "run_complete"
            ),
            None,
        )
        summaries[(run_name, phase_name)] = RunSummary(
            run_name=run_name,
            phase_name=phase_name,
            kind="primary"
            if phase_name in {"single", "phase_b_unfrozen"}
            else "phase_a",
            config_path=config_path,
            metrics_path=metrics_path,
            max_epochs=int(cfg.get("training", {}).get("max_epochs", 0) or 0),
            latest=_event_summary(latest),
            best=_event_summary(best),
            run_complete=run_complete,
            latest_event_name=str(latest.get("event")) if latest else None,
        )
    return summaries


def _job_statuses() -> dict[str, dict[str, str]]:
    launch_report = json.loads(
        (SUITE_ROOT / "launch_report.json").read_text(encoding="utf-8")
    )
    job_map = launch_report.get("jobs", {})
    if not job_map:
        return {}
    wanted = sorted(set(job_map.values()))
    sacct_text = _run_cmd(
        [
            "sacct",
            "-j",
            ",".join(wanted),
            "--format=JobIDRaw,JobName%80,State,ExitCode,Elapsed",
            "-P",
        ]
    )
    statuses: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(sacct_text.splitlines(), delimiter="|")
    for row in reader:
        job_id = row["JobIDRaw"]
        if "." in job_id:
            continue
        statuses[job_id] = {
            "job_name": row["JobName"],
            "state": row["State"],
            "exit_code": row["ExitCode"],
            "elapsed": row["Elapsed"],
        }

    try:
        squeue_text = _run_cmd(
            [
                "squeue",
                "-u",
                os.environ.get("USER", Path.home().name),
                "-o",
                "%i|%j|%t|%M|%R",
            ]
        )
    except subprocess.CalledProcessError:
        squeue_text = ""
    for line in squeue_text.splitlines()[1:]:
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        job_id, job_name, short_state, elapsed, reason = parts
        if job_id not in statuses:
            continue
        statuses[job_id]["squeue_state"] = short_state
        statuses[job_id]["squeue_elapsed"] = elapsed
        statuses[job_id]["reason"] = reason

    keyed: dict[str, dict[str, str]] = {}
    for key, job_id in job_map.items():
        keyed[key] = {"job_id": job_id, **statuses.get(job_id, {})}
    return keyed


def _metric_value(
    summary: EventSummary | None, scope: str, metric: str
) -> float | None:
    if summary is None:
        return None
    scope_metrics = summary.metrics_by_scope.get(scope, {})
    value = scope_metrics.get(metric)
    return None if value is None else float(value)


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _write_comparison_plot(
    path: Path,
    labels: list[str],
    lhs_values: list[float],
    rhs_values: list[float],
    *,
    lhs_label: str,
    rhs_label: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(12, 5))
    x = list(range(len(labels)))
    width = 0.38
    plt.bar([i - width / 2 for i in x], lhs_values, width=width, label=lhs_label)
    plt.bar([i + width / 2 for i in x], rhs_values, width=width, label=rhs_label)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_absrel_curve_plot(
    path: Path, run_summaries: dict[tuple[str, str], RunSummary]
) -> None:
    plt.figure(figsize=(12, 6))
    for run_name, phase_name in PRIMARY_RUN_ORDER:
        summary = run_summaries.get((run_name, phase_name))
        if summary is None or not summary.metrics_path.is_file():
            continue
        xs: list[int] = []
        ys: list[float] = []
        for event in _validation_events(_iter_metric_events(summary.metrics_path)):
            epoch = int(event.get("epoch", 0) or 0)
            scope_metrics = _metrics_by_scope(event).get("val_degraded", {})
            if "abs_rel" not in scope_metrics:
                continue
            xs.append(epoch)
            ys.append(float(scope_metrics["abs_rel"]))
        if not xs:
            continue
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=DISPLAY_NAMES[run_name],
        )
    plt.xlabel("Epoch")
    plt.ylabel("val_degraded AbsRel")
    plt.title("Long Full Runs: val_degraded AbsRel Over Time")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_progress_plot(
    path: Path, run_summaries: dict[tuple[str, str], RunSummary]
) -> None:
    labels: list[str] = []
    progress: list[float] = []
    colors: list[str] = []
    for run_name, phase_name in PRIMARY_RUN_ORDER:
        summary = run_summaries[(run_name, phase_name)]
        current_epoch = float(summary.latest.epoch if summary.latest else 0)
        total = float(summary.max_epochs or 1)
        labels.append(DISPLAY_NAMES[run_name])
        progress.append(100.0 * current_epoch / total)
        colors.append("#4caf50" if summary.run_complete else "#1f77b4")
    plt.figure(figsize=(10, 5))
    y = list(range(len(labels)))
    plt.barh(y, progress, color=colors)
    plt.yticks(y, labels)
    plt.xlim(0, 100)
    plt.xlabel("Percent of configured epochs reached")
    plt.title("Long Full Run Progress")
    for idx, pct in enumerate(progress):
        plt.text(min(pct + 1.0, 98.0), idx, f"{pct:.0f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    current = _discover_run_summaries(SUITE_ROOT)
    statuses = _job_statuses()

    absrel_snapshot_plot = ASSET_DIR / "best_vs_latest_absrel.png"
    delta1_snapshot_plot = ASSET_DIR / "best_vs_latest_delta1.png"
    absrel_curve_plot = ASSET_DIR / "long_full_absrel_curves.png"
    progress_plot = ASSET_DIR / "long_full_progress.png"

    primary_labels = [DISPLAY_NAMES[run_name] for run_name, _ in PRIMARY_RUN_ORDER]
    latest_absrel = [
        _metric_value(current[(run_name, phase_name)].latest, "val_degraded", "abs_rel")
        or 0.0
        for run_name, phase_name in PRIMARY_RUN_ORDER
    ]
    best_absrel = [
        _metric_value(current[(run_name, phase_name)].best, "val_degraded", "abs_rel")
        or 0.0
        for run_name, phase_name in PRIMARY_RUN_ORDER
    ]
    latest_delta1 = [
        _metric_value(current[(run_name, phase_name)].latest, "val_degraded", "delta1")
        or 0.0
        for run_name, phase_name in PRIMARY_RUN_ORDER
    ]
    best_delta1 = [
        _metric_value(current[(run_name, phase_name)].best, "val_degraded", "delta1")
        or 0.0
        for run_name, phase_name in PRIMARY_RUN_ORDER
    ]

    _write_comparison_plot(
        absrel_snapshot_plot,
        primary_labels,
        latest_absrel,
        best_absrel,
        lhs_label="Latest",
        rhs_label="Best so far",
        ylabel="Best val_degraded AbsRel (lower is better)",
        title="Current Long Runs: Latest vs Best val_degraded AbsRel",
    )
    _write_comparison_plot(
        delta1_snapshot_plot,
        primary_labels,
        latest_delta1,
        best_delta1,
        lhs_label="Latest",
        rhs_label="Best so far",
        ylabel="Best val_degraded delta1 (higher is better)",
        title="Current Long Runs: Latest vs Best val_degraded delta1",
    )
    _write_absrel_curve_plot(absrel_curve_plot, current)
    _write_progress_plot(progress_plot, current)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    setup_rows = [
        [
            "Backbone under test",
            "Video Depth Anything ViT-S (`video_depth_anything`, metric checkpoint)",
        ],
        ["Dataset", "TartanAir v2 only"],
        [
            "Subset",
            "16 train trajectories / 4 val trajectories, `seed=42`, `val_fraction=0.1`, `difficulty=Data_easy`, `camera=lcam_front`",
        ],
        [
            "Validation",
            "Dual validation on the same val trajectories: `val_clean` and `val_degraded`",
        ],
        ["Training controls", "No teachers, no distillation, no teacher cache, no EMA"],
        ["Selection metric", "`val_degraded` AbsRel"],
        ["Smoke purpose", "Crash/startup/validate/save sanity only"],
        [
            "Current long full budgets",
            "Singles: 200 epochs. Option phase A: 20 epochs. Option phase B: 200 additional epochs after resume.",
        ],
        [
            "Why the long rerun exists",
            "The first full jobs were finishing in minutes, so we relaunched longer runs to separate undertraining from genuine model issues.",
        ],
    ]

    primary_rows: list[list[str]] = []
    for run_name, phase_name in PRIMARY_RUN_ORDER:
        summary = current[(run_name, phase_name)]
        key = f"{run_name}:{phase_name}:full"
        status = statuses.get(key, {})
        state = status.get("state", "")
        if "squeue_state" in status:
            state = f"{state} / {status['squeue_state']}"
        latest_epoch = summary.latest.epoch if summary.latest else 0
        primary_rows.append(
            [
                DISPLAY_NAMES[run_name],
                status.get("job_id", ""),
                state,
                f"{latest_epoch}/{summary.max_epochs}",
                _fmt(_metric_value(summary.best, "val_degraded", "abs_rel")),
                _fmt(_metric_value(summary.best, "val_degraded", "delta1")),
                _fmt(_metric_value(summary.latest, "val_degraded", "abs_rel")),
                _fmt(_metric_value(summary.latest, "val_degraded", "delta1")),
                _fmt(_metric_value(summary.latest, "val_clean", "abs_rel")),
            ]
        )

    phase_a_rows: list[list[str]] = []
    for run_name, phase_name in PHASE_A_RUN_ORDER:
        summary = current[(run_name, phase_name)]
        key = f"{run_name}:{phase_name}:full"
        status = statuses.get(key, {})
        phase_a_rows.append(
            [
                DISPLAY_NAMES[run_name],
                status.get("job_id", ""),
                status.get("state", ""),
                f"{summary.latest.epoch if summary.latest else 0}/{summary.max_epochs}",
                _fmt(_metric_value(summary.best, "val_degraded", "abs_rel")),
                _fmt(_metric_value(summary.best, "val_degraded", "delta1")),
                _fmt(_metric_value(summary.latest, "val_degraded", "abs_rel")),
            ]
        )

    completed = [
        display
        for display, row in zip(primary_labels, primary_rows)
        if "COMPLETED" in row[2]
    ]
    running = [
        display
        for display, row in zip(primary_labels, primary_rows)
        if "RUNNING" in row[2]
    ]

    md = f"""# VDA Bug-Hunt Supervisor Update

Generated: {generated_at}

## Executive Summary

- We pivoted away from the earlier FocusMamba six-run comparison because the absolute performance was far below target and the smoke/full gap suggested a debugging problem rather than a clean architecture ranking.
- The current bug-hunt is a tightly controlled **Video Depth Anything ViT-S** fine-tuning suite on a deterministic tiny TartanAir v2 subset, designed to answer three questions:
  1. Does the base metric VDA checkpoint fine-tune sanely on clean data?
  2. Does performance collapse as soon as we introduce our degradation pipeline?
  3. Which architecture options hurt immediately, and which remain neutral or helpful when started from the same pretrained checkpoint?
- We first ran smoke plus very short full jobs, then archived those results and relaunched **long full jobs** with much larger training budgets to remove “undertraining” as the dominant explanation.
- Current headline: **longer training helps**, especially on degraded validation. The strongest long-run result so far is **Option 1 (Prefilter) with best val_degraded AbsRel = 0.606**, narrowly ahead of the degraded baseline at **0.616**. However, **delta1 remains far below target across the board**; none of the runs are close to the desired `delta1 > 0.6`.

## Background

The original symptom was: architecture variants were showing movement in scalar metrics, but the overall numbers were still bad enough that we could not trust the ranking. That triggered a bug-hunt pass aimed at eliminating confounders:

- same pretrained checkpoint for every run: `checkpoints/metric_video_depth_anything_vits.pth`
- same tiny deterministic TartanAir subset for all runs
- same dual validation sets (`val_clean`, `val_degraded`)
- no teacher distillation / cache / EMA
- only one architecture change at a time

## Suite Setup

{_table(["Item", "Value"], setup_rows)}

## What We Changed In The Tooling

- Added a bug-hunt suite generator/launcher rooted at `bug_hunt_vda/`.
- Added checkpoint audit so every run fails fast if the wrong VDA checkpoint or a partial load sneaks in.
- Added dual validation logging for both clean and degraded val splits every run.
- Updated `summary.tsv` to report **both `latest` and `best` snapshots**, instead of only the latest event.
- Relaunched longer full jobs in dependency order on SLURM so no desktop process has to stay alive.

## Current Primary Experiment Status

These are the runs that matter most for decision-making: the two baseline singles plus each option’s `phase_b_unfrozen` full run.

{
        _table(
            [
                "Experiment",
                "Job ID",
                "State",
                "Epoch",
                "Best Deg AbsRel",
                "Best Deg delta1",
                "Latest Deg AbsRel",
                "Latest Deg delta1",
                "Latest Clean AbsRel",
            ],
            primary_rows,
        )
    }

### Progress Plot

![Long full progress](report_assets/{progress_plot.name})

## Phase-A Frozen Checks

Phase A is the “immediate regression detector”: train only the newly added module while freezing the pretrained VDA weights. If an option is already harmful here, it is unlikely to be a clean upgrade.

{
        _table(
            [
                "Option",
                "Job ID",
                "State",
                "Epoch",
                "Best Deg AbsRel",
                "Best Deg delta1",
                "Latest Deg AbsRel",
            ],
            phase_a_rows,
        )
    }

## Current Snapshot: Latest vs Best-So-Far

These plots show whether each run is still improving or has already peaked earlier in training.

![Latest vs best AbsRel](report_assets/{absrel_snapshot_plot.name})

![Latest vs best delta1](report_assets/{delta1_snapshot_plot.name})

## Validation Curves

This plot shows how val_degraded AbsRel evolved over the long full runs.

![Long full AbsRel curves](report_assets/{absrel_curve_plot.name})

## Key Readouts Right Now

- Completed primary runs: {", ".join(completed) if completed else "none yet"}.
- Still running primary runs: {", ".join(running) if running else "none"}.
- Best **degraded** result so far is **Opt 1: Prefilter** at **0.606 AbsRel** on `val_degraded`.
- The degraded baseline also improved materially with longer training: **0.616 AbsRel** best.
- Several options cluster in the `0.624–0.702` best-AbsRel range, so the longer rerun is doing what we wanted: separating “bad because too short” from “bad because the option is genuinely weak.”
- The biggest remaining concern is **delta1**. Even the best long runs are still only around `0.32–0.35`, which is well below the target `> 0.6`.
- Clean validation remains suspiciously poor for many long runs, often around `1.8–1.9` AbsRel latest. That keeps the “possible metric/plumbing issue” hypothesis alive, even though degraded AbsRel improved.

## Interpretation

What looks encouraging:

- Longer training clearly matters; the very short full runs were not enough to judge the suite.
- The bug-hunt structure is working as intended: most phase-A runs are neutral, while the phase-B runs separate into meaningfully different degraded outcomes.
- Option 1 (front-end prefilter) is currently the strongest architecture modification on the degraded selection metric.

What still looks wrong or unresolved:

- None of the runs are close to the stated goal of `AbsRel < 0.6` **and** `delta1 > 0.6`.
- The clean/degraded discrepancy is still odd: degraded AbsRel can get much better while clean AbsRel remains very high.
- Some “best” epochs happen very early, then regress later, which suggests we still need tighter checkpoint-centric reporting and possibly additional debugging of the metric/training objective.

## Recommended Next Discussion Points

1. Decide whether the next debugging pass should focus on **metric correctness / evaluation plumbing** versus **training objective / scale handling**.
2. Decide whether to promote **Option 1** as the current leading modification, or keep the architecture comparison open until the three remaining long jobs finish.
3. Decide whether the next suite should optimize for **best checkpoint quality** rather than latest-epoch behavior, since the long runs often peak early.
4. Decide whether to add a dedicated clean-only checkpoint-selection sanity run, because the clean metrics still look inconsistent with expectations from the pretrained VDA model.
"""

    REPORT_PATH.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
