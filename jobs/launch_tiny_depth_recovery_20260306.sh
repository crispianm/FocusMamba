#!/bin/bash
# Submit the focused tiny metric-depth recovery runs for March 6.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
CFG_DIR="${CFG_DIR:-configs/experiments/generated/tiny_depth_recovery_20260306}"

cd "$PROJECT_DIR"
mkdir -p logs

resume_a2="${PROJECT_DIR}/checkpoints/tiny_dual_curriculum_20260305/02_stageA_rel_unbounded_smoke4/best.pt"
run1_best="${PROJECT_DIR}/checkpoints/tiny_depth_recovery_20260306/01_a2_gt_bridge_frozen_10ep/best.pt"

if [[ ! -f "$resume_a2" ]]; then
  echo "ERROR: missing Stage-A checkpoint: $resume_a2"
  exit 1
fi

submit() {
  local job_name="$1"
  local config_path="$2"
  local resume_from="${3:-}"
  local dependency="${4:-}"

  if [[ ! -f "$config_path" ]]; then
    echo "ERROR: missing config: $config_path"
    exit 1
  fi
  if [[ -n "$resume_from" && ! -f "$resume_from" && -z "$dependency" ]]; then
    echo "ERROR: missing resume checkpoint: $resume_from"
    exit 1
  fi

  local -a sbatch_args=(
    --parsable
    --job-name="$job_name"
    --output="./logs/${job_name}_%j.out"
    --error="./logs/${job_name}_%j.err"
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${config_path},RESET_CHECKPOINTS=1,RESUME_FROM=${resume_from}"
  )
  if [[ -n "$dependency" ]]; then
    sbatch_args+=(--dependency="afterok:${dependency}")
  fi

  sbatch "${sbatch_args[@]}" "$RUNNER"
}

echo "Submitting tiny depth recovery suite from $CFG_DIR"

jid_gt="$(submit "tnyR_gt10" "${CFG_DIR}/01_a2_gt_bridge_frozen_10ep.yaml" "$resume_a2")"
jid_unf="$(submit "tnyR_unf10" "${CFG_DIR}/02_a2_gt_bridge_unfreeze_e3_10ep.yaml" "$resume_a2")"
jid_ctl="$(submit "tnyR_ctl4" "${CFG_DIR}/04_control_legacy_vda_tiny_518x16_4ep.yaml" "")"
jid_vda="$(submit "tnyR_vda10" "${CFG_DIR}/03_a2_vda_teacher_after_gt_10ep.yaml" "$run1_best" "$jid_gt")"

echo "Submitted jobs:"
echo "  ${jid_gt}  tnyR_gt10   ${CFG_DIR}/01_a2_gt_bridge_frozen_10ep.yaml"
echo "  ${jid_unf} tnyR_unf10  ${CFG_DIR}/02_a2_gt_bridge_unfreeze_e3_10ep.yaml"
echo "  ${jid_ctl} tnyR_ctl4   ${CFG_DIR}/04_control_legacy_vda_tiny_518x16_4ep.yaml"
echo "  ${jid_vda} tnyR_vda10  ${CFG_DIR}/03_a2_vda_teacher_after_gt_10ep.yaml  (afterok:${jid_gt})"

echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
