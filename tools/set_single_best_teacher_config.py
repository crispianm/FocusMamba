#!/usr/bin/env python
"""Rewrite a teacher-sweep config to use the best audited teacher."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", required=True, help="Path to teacher_summary.csv from audit_teacher_outputs.py.")
    parser.add_argument("--config", required=True, help="Config YAML to rewrite in-place.")
    parser.add_argument(
        "--metric",
        default="abs_rel",
        help="Audit column used to choose the best teacher. Lower is better.",
    )
    return parser.parse_args()


def load_best_teacher(audit_csv: Path, metric: str) -> str:
    with audit_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in audit CSV: {audit_csv}")
    if metric not in rows[0]:
        raise KeyError(f"Metric column {metric!r} not found in {audit_csv}")

    best_row = min(rows, key=lambda row: float(row[metric]))
    teacher = best_row.get("teacher")
    if not teacher:
        raise ValueError(f"Missing teacher column in {audit_csv}")
    return teacher


def rewrite_config(config_path: Path, teacher_name: str) -> None:
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    enabled_teachers = [t for t in cfg.get("teachers", []) if t.get("name") == teacher_name]
    if not enabled_teachers:
        raise ValueError(f"Teacher {teacher_name!r} is not present in {config_path}")

    cfg["teachers"] = enabled_teachers
    distill_cfg = cfg.setdefault("distillation", {})
    distill_cfg["teachers"] = [{"name": teacher_name, "weight": 1.0}]
    distill_cfg["strategy"] = "calibrated_aggregate"

    with config_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    audit_csv = Path(args.audit_csv)
    config_path = Path(args.config)
    best_teacher = load_best_teacher(audit_csv, args.metric)
    rewrite_config(config_path, best_teacher)
    print(f"Updated {config_path} to use best audited teacher: {best_teacher}")


if __name__ == "__main__":
    main()
