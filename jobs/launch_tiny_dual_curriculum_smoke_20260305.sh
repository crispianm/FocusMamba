#!/bin/bash
# Submit short tiny dual-head DINOv3 curriculum ablations.
# - Stage A jobs are <= 4 epochs.
# - Stage B jobs are <= 3 epochs and depend on Stage A completion.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
CFG_DIR="${CFG_DIR:-configs/experiments/generated/tiny_dual_curriculum_20260305}"

cd "$PROJECT_DIR"
mkdir -p logs

submit() {
  local job_name="$1"
  local config_path="$2"
  local dependency="${3:-}"
  local resume_from="${4:-}"

  local -a cmd=(
    sbatch
    --parsable
    --job-name="$job_name"
    --output="./logs/${job_name}_%j.out"
    --error="./logs/${job_name}_%j.err"
  )

  if [[ -n "$dependency" ]]; then
    cmd+=(--dependency="afterok:${dependency}")
  fi

  local export_vars="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${config_path},RESET_CHECKPOINTS=1"
  if [[ -n "$resume_from" ]]; then
    export_vars="${export_vars},RESUME_FROM=${resume_from}"
  fi
  cmd+=(--export="$export_vars" "$RUNNER")

  "${cmd[@]}"
}

echo "Submitting tiny dual-head curriculum smoke suite from $CFG_DIR"

cfg_a1="${CFG_DIR}/01_stageA_rel_bounded_smoke4.yaml"
cfg_a2="${CFG_DIR}/02_stageA_rel_unbounded_smoke4.yaml"
cfg_a3="${CFG_DIR}/03_stageA_rel_bounded_no_clamp_smoke4.yaml"
cfg_m0="${CFG_DIR}/04_direct_metric_smoke4.yaml"
cfg_b1="${CFG_DIR}/11_stageB_metric_from_01_smoke3.yaml"
cfg_b2="${CFG_DIR}/12_stageB_metric_from_02_smoke3.yaml"
cfg_b3="${CFG_DIR}/13_stageB_metric_from_03_smoke3.yaml"

for cfg in "$cfg_a1" "$cfg_a2" "$cfg_a3" "$cfg_m0" "$cfg_b1" "$cfg_b2" "$cfg_b3"; do
  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: missing config: $cfg"
    exit 1
  fi
done

j_a1="$(submit tnyA_rel_bd4 "$cfg_a1")"
j_a2="$(submit tnyA_rel_ub4 "$cfg_a2")"
j_a3="$(submit tnyA_rel_nc4 "$cfg_a3")"
j_m0="$(submit tnyM_dir_4 "$cfg_m0")"

resume_a1="${PROJECT_DIR}/checkpoints/tiny_dual_curriculum_20260305/01_stageA_rel_bounded_smoke4/best.pt"
resume_a2="${PROJECT_DIR}/checkpoints/tiny_dual_curriculum_20260305/02_stageA_rel_unbounded_smoke4/best.pt"
resume_a3="${PROJECT_DIR}/checkpoints/tiny_dual_curriculum_20260305/03_stageA_rel_bounded_no_clamp_smoke4/best.pt"

j_b1="$(submit tnyB_fromA1_3 "$cfg_b1" "$j_a1" "$resume_a1")"
j_b2="$(submit tnyB_fromA2_3 "$cfg_b2" "$j_a2" "$resume_a2")"
j_b3="$(submit tnyB_fromA3_3 "$cfg_b3" "$j_a3" "$resume_a3")"

echo "Submitted jobs:"
echo "  StageA bounded      : $j_a1"
echo "  StageA unbounded    : $j_a2"
echo "  StageA no-clamp     : $j_a3"
echo "  Direct metric       : $j_m0"
echo "  StageB <- bounded   : $j_b1 (afterok:$j_a1)"
echo "  StageB <- unbounded : $j_b2 (afterok:$j_a2)"
echo "  StageB <- no-clamp  : $j_b3 (afterok:$j_a3)"
echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.18j %.8T %.10M %.6D %R'
