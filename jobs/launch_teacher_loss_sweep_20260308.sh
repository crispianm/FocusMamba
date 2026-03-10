#!/bin/bash
# Submit teacher-loss sweep runs for degraded streaming VDA-S.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
CFG_DIR="${CFG_DIR:-configs/experiments/generated/teacher_loss_sweep_20260308}"

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

  if [[ ! -f "$config_path" ]]; then
    echo "ERROR: missing config: $config_path"
    exit 1
  fi

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

echo "Submitting teacher-loss sweep from $CFG_DIR"

jid_bugfix="$(submit "tBug" "${CFG_DIR}/01_teacher_bugfix_only.yaml")"
jid_agg="$(submit "tAgg" "${CFG_DIR}/02_teacher_calibrated_agg.yaml")"
jid_het="$(submit "tHet" "${CFG_DIR}/03_teacher_hetero_mtkd.yaml")"
jid_anchor="$(submit "tAnc" "${CFG_DIR}/04_teacher_anchor_vda.yaml")"
jid_single="$(submit "tSng" "${CFG_DIR}/05_teacher_single_best_calibrated.yaml")"
jid_gate="$(submit "tGtH" "${CFG_DIR}/06_teacher_hetero_mtkd_plus_gate.yaml")"

echo "Submitted jobs:"
echo "  ${jid_bugfix}  tBug  ${CFG_DIR}/01_teacher_bugfix_only.yaml"
echo "  ${jid_agg}     tAgg  ${CFG_DIR}/02_teacher_calibrated_agg.yaml"
echo "  ${jid_het}     tHet  ${CFG_DIR}/03_teacher_hetero_mtkd.yaml"
echo "  ${jid_anchor}  tAnc  ${CFG_DIR}/04_teacher_anchor_vda.yaml"
echo "  ${jid_single}  tSng  ${CFG_DIR}/05_teacher_single_best_calibrated.yaml"
echo "  ${jid_gate}    tGtH  ${CFG_DIR}/06_teacher_hetero_mtkd_plus_gate.yaml"

echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
