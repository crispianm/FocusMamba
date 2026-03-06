"""Logging and run-artifact helpers for training jobs.

This module keeps console/file logging, config snapshots, run metadata, and
structured metric emission in one place so the training entrypoint can stay
focused on orchestration.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from rich.console import Console
    from rich.logging import RichHandler
except Exception:  # pragma: no cover - optional dependency
    Console = None
    RichHandler = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def setup_logger(
    name: str,
    *,
    debug: bool = False,
    log_file: str | Path | None = None,
    use_rich: bool = True,
) -> logging.Logger:
    """Create a console+file logger with sane defaults."""
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    _close_handlers(logger)

    if use_rich and RichHandler is not None and Console is not None:
        console = Console(stderr=False, soft_wrap=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path.resolve(), mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

    logging.captureWarnings(True)
    return logger


@dataclass(frozen=True)
class RunArtifacts:
    log_dir: Path
    verbose_log_file: Path
    metrics_file: Path
    metadata_file: Path
    config_snapshot_file: Path


def resolve_run_artifacts(train_cfg: Mapping[str, Any], *, default_log_dir: str) -> RunArtifacts:
    log_dir = Path(train_cfg.get("log_dir", default_log_dir))
    verbose_log_file = Path(train_cfg.get("verbose_log_file", log_dir / "train_verbose.log"))
    metrics_file = Path(train_cfg.get("metrics_jsonl_file", log_dir / "metrics.jsonl"))
    metadata_file = Path(train_cfg.get("run_metadata_file", log_dir / "run_metadata.json"))
    config_snapshot_file = Path(train_cfg.get("config_snapshot_file", log_dir / "config_snapshot.yaml"))
    return RunArtifacts(
        log_dir=log_dir,
        verbose_log_file=verbose_log_file,
        metrics_file=metrics_file,
        metadata_file=metadata_file,
        config_snapshot_file=config_snapshot_file,
    )


def write_run_metadata(
    *,
    artifacts: RunArtifacts,
    cfg: Mapping[str, Any],
    args: Mapping[str, Any],
    project_root: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts.log_dir.mkdir(parents=True, exist_ok=True)
    artifacts.config_snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    artifacts.metadata_file.parent.mkdir(parents=True, exist_ok=True)

    config_text = yaml.safe_dump(dict(cfg), sort_keys=False)
    artifacts.config_snapshot_file.write_text(config_text)

    train_cfg = dict(cfg.get("training", {}) or {})
    loss_cfg = dict(cfg.get("loss", {}) or {})
    metadata: dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "project_root": str(project_root),
        "python": sys.version.split()[0],
        "command": list(sys.argv),
        "config_path": args.get("config"),
        "resume_path": args.get("resume"),
        "device_arg": args.get("device"),
        "debug": bool(args.get("debug", False)),
        "verbose": bool(args.get("verbose", False)),
        "log_dir": str(artifacts.log_dir),
        "verbose_log_file": str(artifacts.verbose_log_file),
        "metrics_file": str(artifacts.metrics_file),
        "config_snapshot_file": str(artifacts.config_snapshot_file),
        "target_mode": str(loss_cfg.get("training_target", "metric")).lower(),
        "selection_metric": train_cfg.get("selection_metric"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "world_size_env": os.environ.get("WORLD_SIZE"),
        "local_rank_env": os.environ.get("LOCAL_RANK"),
        "git_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "git_branch": _git_output(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
    }
    if extra:
        metadata.update(dict(extra))

    artifacts.metadata_file.write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True) + "\n")
    return metadata


class JsonlMetricLogger:
    """Append-only structured event logger for runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log_event(self, event: str, **fields: Any) -> None:
        payload = {
            "ts": _utc_now_iso(),
            "event": event,
            **{key: _json_ready(value) for key, value in fields.items()},
        }
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
