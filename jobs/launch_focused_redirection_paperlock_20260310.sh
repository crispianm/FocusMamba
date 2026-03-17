#!/bin/bash
# Generate, audit, and submit the focused-redirection paper-lock suite.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
SUITE_NAME="${SUITE_NAME:-focused_redirection_paperlock_20260310}"
PROFILE="${PROFILE:-promote}"
GENERATOR="${GENERATOR:-tools/generate_focused_redirection_suite.py}"
AUDITOR="${AUDITOR:-tools/audit_focused_redirection_launch.py}"
CFG_DIR="${CFG_DIR:-configs/experiments/generated/${SUITE_NAME}}"
AUDIT_TSV="${AUDIT_TSV:-${CFG_DIR}/launch_audit.tsv}"
JOBS_TSV="${JOBS_TSV:-${CFG_DIR}/launch_jobs.tsv}"
SUBMIT_JOBS="${SUBMIT_JOBS:-1}"

cd "$PROJECT_DIR"

"${PROJECT_DIR}/.venv/bin/python" "$GENERATOR" \
  --project-root "$PROJECT_DIR" \
  --suite-name "$SUITE_NAME" \
  --profile "$PROFILE"

"${PROJECT_DIR}/.venv/bin/python" "$AUDITOR" \
  --project-root "$PROJECT_DIR" \
  --candidate-suite "$CFG_DIR" \
  --audit-tsv "$AUDIT_TSV" \
  --launch-tsv "$JOBS_TSV"

echo
echo "Audit summary:"
awk -F'\t' '
  NR == 1 { next }
  { counts[$1]++ }
  END {
    for (k in counts) {
      printf "  %s\t%d\n", k, counts[k]
    }
  }
' "$AUDIT_TSV" | sort

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

if [[ "$SUBMIT_JOBS" != "1" ]]; then
  echo
  echo "SUBMIT_JOBS=$SUBMIT_JOBS -> audit only. Filtered jobs are in $JOBS_TSV"
  exit 0
fi

if [[ ! -f "$JOBS_TSV" ]]; then
  echo "ERROR: missing launch jobs file: $JOBS_TSV" >&2
  exit 1
fi

submitted=0
echo
echo "Submitting launch-now jobs from $JOBS_TSV"
while IFS=$'\t' read -r job_name config_path; do
  job_name="${job_name//$'\r'/}"
  config_path="${config_path//$'\r'/}"
  if [[ "$job_name" == "job_name" ]]; then
    continue
  fi
  jid="$(submit "$job_name" "$config_path")"
  printf '%s\t%s\t%s\n' "$jid" "$job_name" "$config_path"
  submitted=$((submitted + 1))
done < "$JOBS_TSV"

echo
echo "Submitted jobs: $submitted"
echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
