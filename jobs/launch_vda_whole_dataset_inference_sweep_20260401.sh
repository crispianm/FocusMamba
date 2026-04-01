#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_vda_degraded_benchmark.sh}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/vda_degraded_benchmark/whole_dataset_inference_sweep_${STAMP}}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

submit() {
  local job_name="$1"
  local config_path="$2"
  local checkpoint_path="$3"
  local checkpoint_label="$4"
  local inference_size="$5"
  local output_dir="$6"

  sbatch \
    --parsable \
    --job-name="$job_name" \
    --output="${output_dir}/${job_name}_%j.out" \
    --error="${output_dir}/${job_name}_%j.err" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${config_path},CHECKPOINT_PATH=${checkpoint_path},CHECKPOINT_LABEL=${checkpoint_label},CHECKPOINT_VARIANT=small,NUM_EVAL_CLIPS=32,NUM_DISPLAY_CLIPS=2,IMAGE_SIZE=392,INFERENCE_SIZE=${inference_size},OUTPUT_DIR=${output_dir},OUTPUT_PREFIX=${checkpoint_label}_degraded_eval" \
    "$RUNNER"
}

echo "Submitting degraded TartanAir inference-size sweep for whole-dataset VDA checkpoints"

qg_336_out="${OUTPUT_ROOT}/quality_gate_336"
dg_336_out="${OUTPUT_ROOT}/degradation_conditioned_336"
qg_320_out="${OUTPUT_ROOT}/quality_gate_320"
dg_320_out="${OUTPUT_ROOT}/degradation_conditioned_320"
mkdir -p "$qg_336_out" "$dg_336_out" "$qg_320_out" "$dg_320_out"

jid_qg_336="$(
  submit \
    "vdaQ336" \
    "runs/vda_bug_hunt_promotions_20260319/04_quality_gate_whole_dataset/config_snapshot.yaml" \
    "checkpoints/vda_bug_hunt_promotions_20260319/04_quality_gate_whole_dataset/best.pt" \
    "quality_gate_whole_dataset_336" \
    "336" \
    "$qg_336_out"
)"
jid_dg_336="$(
  submit \
    "vdaD336" \
    "runs/vda_bug_hunt_promotions_20260319/07_degradation_conditioned_gate_whole_dataset/config_snapshot.yaml" \
    "checkpoints/vda_bug_hunt_promotions_20260319/07_degradation_conditioned_gate_whole_dataset/best.pt" \
    "degradation_conditioned_whole_dataset_336" \
    "336" \
    "$dg_336_out"
)"
jid_qg_320="$(
  submit \
    "vdaQ320" \
    "runs/vda_bug_hunt_promotions_20260319/04_quality_gate_whole_dataset/config_snapshot.yaml" \
    "checkpoints/vda_bug_hunt_promotions_20260319/04_quality_gate_whole_dataset/best.pt" \
    "quality_gate_whole_dataset_320" \
    "320" \
    "$qg_320_out"
)"
jid_dg_320="$(
  submit \
    "vdaD320" \
    "runs/vda_bug_hunt_promotions_20260319/07_degradation_conditioned_gate_whole_dataset/config_snapshot.yaml" \
    "checkpoints/vda_bug_hunt_promotions_20260319/07_degradation_conditioned_gate_whole_dataset/best.pt" \
    "degradation_conditioned_whole_dataset_320" \
    "320" \
    "$dg_320_out"
)"

echo "Submitted jobs:"
echo "  ${jid_qg_336}  vdaQ336  quality-gate whole-dataset benchmark @ 336"
echo "  ${jid_dg_336}  vdaD336  degradation-conditioned whole-dataset benchmark @ 336"
echo "  ${jid_qg_320}  vdaQ320  quality-gate whole-dataset benchmark @ 320"
echo "  ${jid_dg_320}  vdaD320  degradation-conditioned whole-dataset benchmark @ 320"

echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
