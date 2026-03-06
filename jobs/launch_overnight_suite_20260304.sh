#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
cd "$PROJECT_DIR"
mkdir -p logs
STAMP="$(date -u +%Y%m%d_%H%M%S)"
SUBMIT_LOG="logs/overnight_20260304_submit_${STAMP}.txt"

entries=(
  "ovn_tiny_full_base|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/01_ovn_tiny_full_base.yaml|Tiny full-data baseline distill (no MTKD)."
  "ovn_tiny_full_mtkd|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/02_ovn_tiny_full_mtkd.yaml|Tiny full-data MTKD default."
  "ovn_small_full_base|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/03_ovn_small_full_base.yaml|Small full-data baseline distill (no MTKD)."
  "ovn_small_full_mtkd|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/04_ovn_small_full_mtkd.yaml|Small full-data MTKD default."
  "ovn_tiny_quick_alpha03|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/05_ovn_tiny_quick_alpha03.yaml|Tiny quick: MTKD alpha 0.3."
  "ovn_tiny_quick_alpha10|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/06_ovn_tiny_quick_alpha10.yaml|Tiny quick: MTKD alpha 1.0."
  "ovn_tiny_quick_wav0|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/07_ovn_tiny_quick_wav0.yaml|Tiny quick: MTKD wavelet disabled."
  "ovn_tiny_quick_temp0|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/08_ovn_tiny_quick_temp0.yaml|Tiny quick: MTKD temporal disabled."
  "ovn_tiny_quick_conf0|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/09_ovn_tiny_quick_conf0.yaml|Tiny quick: confidence weighting off."
  "ovn_tiny_quick_dw1|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/10_ovn_tiny_quick_dw1.yaml|Tiny quick: distillation weight 1.0."
  "ovn_tiny_quick_lsi07|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/11_ovn_tiny_quick_lsi07.yaml|Tiny quick: SI-log lambda 0.7."
  "ovn_tiny_quick_res336|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/12_ovn_tiny_quick_res336.yaml|Tiny quick: lower resolution 336 for speed/perf tradeoff."
  "ovn_small_quick_conf0|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/13_ovn_small_quick_conf0.yaml|Small quick: confidence weighting off."
  "ovn_small_quick_dw1|jobs/run_train_vda_small_scratch_distill.sh|configs/experiments/generated/overnight_20260304/14_ovn_small_quick_dw1.yaml|Small quick: distillation weight 1.0."
  "ovn_fm_quick_softplus|jobs/run_train_focusmamba_v2_distill_50ep.sh|configs/experiments/generated/overnight_20260304/15_ovn_fm_quick_softplus.yaml|FocusMamba quick with softplus output."
  "ovn_fm_quick_relu|jobs/run_train_focusmamba_v2_distill_50ep.sh|configs/experiments/generated/overnight_20260304/16_ovn_fm_quick_relu.yaml|FocusMamba quick with relu output."
)

echo "Submitting ${#entries[@]} jobs..." | tee "$SUBMIT_LOG"
echo "timestamp_utc=${STAMP}" | tee -a "$SUBMIT_LOG"

for e in "${entries[@]}"; do
  IFS="|" read -r run_name runner_script config_path note <<< "$e"
  cmd=(sbatch --parsable --job-name="$run_name" --export=ALL,CONFIG="$config_path",RESET_CHECKPOINTS=1 "$runner_script")
  echo "CMD: ${cmd[*]}" | tee -a "$SUBMIT_LOG"
  job_id="$(${cmd[@]})"
  echo "${job_id} | ${run_name} | ${config_path} | ${note}" | tee -a "$SUBMIT_LOG"
done

echo "" | tee -a "$SUBMIT_LOG"
echo "Active queue:" | tee -a "$SUBMIT_LOG"
squeue -u $USER -o '%.18i %.9P %.12j %.8T %.10M %.6D %R' | tee -a "$SUBMIT_LOG"
echo "Submit log: $SUBMIT_LOG"
