#!/bin/bash
# Submit Stage-B recovery jobs for tiny dual-head curriculum.
# - Resumes model weights from Stage-A best checkpoints.
# - Resets per-run checkpoint dirs to avoid stale optimizer/scheduler state.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
CFG_DIR="${CFG_DIR:-configs/experiments/generated/tiny_stageB_recovery_20260305}"

cd "$PROJECT_DIR"
mkdir -p logs

resume_a1="${PROJECT_DIR}/checkpoints/tiny_dual_curriculum_20260305/01_stageA_rel_bounded_smoke4/best.pt"
resume_a2="${PROJECT_DIR}/checkpoints/tiny_dual_curriculum_20260305/02_stageA_rel_unbounded_smoke4/best.pt"

if [[ ! -f "$resume_a1" ]]; then
  echo "ERROR: missing Stage-A checkpoint: $resume_a1"
  exit 1
fi
if [[ ! -f "$resume_a2" ]]; then
  echo "ERROR: missing Stage-A checkpoint: $resume_a2"
  exit 1
fi

submit() {
  local job_name="$1"
  local config_path="$2"
  local resume_from="$3"

  if [[ ! -f "$config_path" ]]; then
    echo "ERROR: missing config: $config_path"
    exit 1
  fi

  sbatch \
    --parsable \
    --job-name="$job_name" \
    --output="./logs/${job_name}_%j.out" \
    --error="./logs/${job_name}_%j.err" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${config_path},RESET_CHECKPOINTS=1,RESUME_FROM=${resume_from}" \
    "$RUNNER"
}

echo "Submitting Stage-B recovery suite from $CFG_DIR"

declare -a JOB_SPECS=(
  "tnyB2_vda6|${CFG_DIR}/01_a2_vda_like_frozen_backbone_6ep.yaml|${resume_a2}"
  "tnyB2_cal6|${CFG_DIR}/02_a2_calibrator_only_6ep.yaml|${resume_a2}"
  "tnyB2_mtk6|${CFG_DIR}/03_a2_frozen_backbone_light_mtkd_6ep.yaml|${resume_a2}"
  "tnyB2_unf6|${CFG_DIR}/04_a2_unfrozen_lowlr_abs_6ep.yaml|${resume_a2}"
  "tnyB2_gt6|${CFG_DIR}/05_a2_gt_only_metric_bridge_6ep.yaml|${resume_a2}"
  "tnyB2_cfw6|${CFG_DIR}/06_a2_vda_like_conf_on_6ep.yaml|${resume_a2}"
  "tnyB1_vda6|${CFG_DIR}/07_a1_vda_like_frozen_backbone_6ep.yaml|${resume_a1}"
  "tnyB1_cal6|${CFG_DIR}/08_a1_calibrator_only_6ep.yaml|${resume_a1}"
  "tnyB2_vda12|${CFG_DIR}/09_a2_vda_like_frozen_backbone_12ep.yaml|${resume_a2}"
  "tnyB2_cal12|${CFG_DIR}/10_a2_calibrator_only_12ep.yaml|${resume_a2}"
)

declare -a submitted=()
for spec in "${JOB_SPECS[@]}"; do
  IFS='|' read -r job_name cfg resume <<< "$spec"
  jid="$(submit "$job_name" "$cfg" "$resume")"
  submitted+=("${jid}|${job_name}|${cfg}")
done

echo "Submitted jobs:"
for item in "${submitted[@]}"; do
  IFS='|' read -r jid jn cfg <<< "$item"
  echo "  ${jid}  ${jn}  ${cfg}"
done

echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
