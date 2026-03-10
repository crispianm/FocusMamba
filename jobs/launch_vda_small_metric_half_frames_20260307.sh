#!/bin/bash
# Submit ACM MM VDA-small streaming ablation runs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
CFG_DIR="${CFG_DIR:-configs/experiments/generated/acmmm_vda_streaming_metric_20260307_half_frames}"

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

student_ckpt="${PROJECT_DIR}/checkpoints/metric_video_depth_anything_vits.pth"

if [[ ! -f "$student_ckpt" ]]; then
  echo "ERROR: missing pretrained student checkpoint: $student_ckpt"
  exit 1
fi

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

echo "Submitting ACM MM VDA-small half-frame overnight suite from $CFG_DIR"

jid_offline="$(submit "hOff" "${CFG_DIR}/01_offline_clean.yaml")"
jid_stream="$(submit "hStr" "${CFG_DIR}/02_stream_clean.yaml")"
jid_degraded="$(submit "hDeg" "${CFG_DIR}/03_stream_degraded.yaml")"
jid_r4="$(submit "hR04" "${CFG_DIR}/04_stream_degraded_reset4.yaml")"
jid_r8="$(submit "hR08" "${CFG_DIR}/05_stream_degraded_reset8.yaml")"
jid_k4="$(submit "hK04" "${CFG_DIR}/06_stream_degraded_shortmem4.yaml")"
jid_gate="$(submit "hGate" "${CFG_DIR}/07_stream_degraded_gated.yaml")"
jid_pref="$(submit "hDegP" "${CFG_DIR}/08_stream_degraded_prefilter.yaml")"
jid_gate_pref="$(submit "hGtPf" "${CFG_DIR}/09_stream_degraded_gated_prefilter.yaml")"

echo "Submitted jobs:"
echo "  ${jid_offline}  hOff    ${CFG_DIR}/01_offline_clean.yaml"
echo "  ${jid_stream}   hStr    ${CFG_DIR}/02_stream_clean.yaml"
echo "  ${jid_degraded} hDeg    ${CFG_DIR}/03_stream_degraded.yaml"
echo "  ${jid_r4}       hR04    ${CFG_DIR}/04_stream_degraded_reset4.yaml"
echo "  ${jid_r8}       hR08    ${CFG_DIR}/05_stream_degraded_reset8.yaml"
echo "  ${jid_k4}       hK04    ${CFG_DIR}/06_stream_degraded_shortmem4.yaml"
echo "  ${jid_gate}     hGate   ${CFG_DIR}/07_stream_degraded_gated.yaml"
echo "  ${jid_pref}     hDegP   ${CFG_DIR}/08_stream_degraded_prefilter.yaml"
echo "  ${jid_gate_pref} hGtPf  ${CFG_DIR}/09_stream_degraded_gated_prefilter.yaml"

echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
