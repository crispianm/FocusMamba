#!/bin/bash
# Submit the deadline-driven degraded streaming rescue sweep.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
CFG_DIR="${CFG_DIR:-configs/experiments/generated/deadline_rescue_sweep_20260309}"
JOBS_TSV="${JOBS_TSV:-${CFG_DIR}/jobs.tsv}"

cd "$PROJECT_DIR"

resolve_run_log_dir() {
  local config_path="$1"
  "${PROJECT_DIR}/.venv/bin/python" - <<'PY' "$config_path" "$PROJECT_DIR"
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
project_dir = Path(sys.argv[2])
with config_path.open("r") as f:
    cfg = yaml.safe_load(f) or {}
log_dir = Path(cfg.get("training", {}).get("log_dir", f"runs/{config_path.stem}"))
if not log_dir.is_absolute():
    log_dir = project_dir / log_dir
print(log_dir / "logs")
PY
}

submit() {
  local job_name="$1"
  local config_path="$2"
  local run_log_dir

  if [[ ! -f "$config_path" ]]; then
    echo "ERROR: missing config: $config_path" >&2
    exit 1
  fi

  run_log_dir="$(resolve_run_log_dir "$config_path")"
  mkdir -p "$run_log_dir"

  sbatch \
    --parsable \
    --job-name="$job_name" \
    --output="${run_log_dir}/${job_name}_%j.out" \
    --error="${run_log_dir}/${job_name}_%j.err" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${config_path},RESET_CHECKPOINTS=1,RESUME_FROM=" \
    "$RUNNER"
}

if [[ ! -f "$JOBS_TSV" ]]; then
  echo "ERROR: missing jobs.tsv: $JOBS_TSV" >&2
  exit 1
fi

echo "Submitting deadline rescue sweep from $CFG_DIR"

while IFS=$'\t' read -r job_name config_path; do
  job_name="${job_name//$'\r'/}"
  config_path="${config_path//$'\r'/}"
  if [[ "$job_name" == "job_name" ]]; then
    continue
  fi
  jid="$(submit "$job_name" "$config_path")"
  printf '%s\t%s\t%s\n' "$jid" "$job_name" "$config_path"
done < "$JOBS_TSV"

echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
