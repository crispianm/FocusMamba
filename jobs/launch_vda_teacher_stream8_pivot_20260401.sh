#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
BENCH_RUNNER="${BENCH_RUNNER:-jobs/run_vda_degraded_benchmark.sh}"
CFG_DIR="${CFG_DIR:-configs/experiments/generated/vda_teacher_stream8_pivot_20260401}"
BENCH_ROOT="${BENCH_ROOT:-runs/vda_degraded_benchmark/teacher_stream8_pivot_$(date -u +%Y%m%d_%H%M%S)}"

cd "$PROJECT_DIR"

mkdir -p "$CFG_DIR" "$BENCH_ROOT"

"${PROJECT_DIR}/.venv/bin/python" - <<'PY' "$PROJECT_DIR" "$CFG_DIR"
from pathlib import Path
import sys
import yaml

project_dir = Path(sys.argv[1])
cfg_dir = Path(sys.argv[2])

bases = {
    "01_quality_gate_anchor_stream8.yaml": project_dir / "configs/experiments/generated/vda_teacher_whole_dataset_20260401/02_quality_gate_anchor_vda.yaml",
    "02_degradation_conditioned_anchor_stream8.yaml": project_dir / "configs/experiments/generated/vda_teacher_whole_dataset_20260401/04_degradation_conditioned_gate_anchor_vda.yaml",
}

for name, base_path in bases.items():
    cfg = yaml.safe_load(base_path.read_text()) or {}
    run_stem = Path(name).stem
    run_dir = f"runs/vda_teacher_stream8_pivot_20260401/{run_stem}"
    ckpt_dir = f"checkpoints/vda_teacher_stream8_pivot_20260401/{run_stem}"

    model = cfg.setdefault("model", {})
    model["mode"] = "streaming_emulated"
    model["num_frames"] = 8
    model["stream_max_cache_len"] = 8
    model["stream_reset_interval"] = 0

    data = cfg.setdefault("data", {})
    data["train_num_frames"] = 8
    data["val_num_frames"] = 8

    training = cfg.setdefault("training", {})
    training["log_dir"] = run_dir
    training["checkpoint_dir"] = ckpt_dir
    training["verbose_log_file"] = f"{run_dir}/train_verbose.log"
    training["metrics_jsonl_file"] = f"{run_dir}/metrics.jsonl"
    training["run_metadata_file"] = f"{run_dir}/run_metadata.json"
    training["config_snapshot_file"] = f"{run_dir}/config_snapshot.yaml"

    (cfg_dir / name).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

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

submit_train() {
  local job_name="$1"
  local config_path="$2"
  local run_log_dir

  run_log_dir="$(resolve_run_log_dir "$config_path")"
  mkdir -p "$run_log_dir"

  sbatch \
    --parsable \
    --job-name="$job_name" \
    --time="24:00:00" \
    --output="${run_log_dir}/${job_name}_%j.out" \
    --error="${run_log_dir}/${job_name}_%j.err" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${config_path},RESET_CHECKPOINTS=1,RESUME_FROM=" \
    "$RUNNER"
}

submit_bench() {
  local job_name="$1"
  local dependency_job="$2"
  local config_path="$3"
  local checkpoint_path="$4"
  local checkpoint_label="$5"
  local output_dir="$6"

  sbatch \
    --parsable \
    --dependency="afterok:${dependency_job}" \
    --job-name="$job_name" \
    --output="${output_dir}/${job_name}_%j.out" \
    --error="${output_dir}/${job_name}_%j.err" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${config_path},CHECKPOINT_PATH=${checkpoint_path},CHECKPOINT_LABEL=${checkpoint_label},CHECKPOINT_VARIANT=small,MODE=streaming_emulated,NUM_EVAL_CLIPS=32,NUM_DISPLAY_CLIPS=2,IMAGE_SIZE=392,OUTPUT_DIR=${output_dir},OUTPUT_PREFIX=${checkpoint_label}_degraded_eval" \
    "$BENCH_RUNNER"
}

cfg_q="${CFG_DIR}/01_quality_gate_anchor_stream8.yaml"
cfg_d="${CFG_DIR}/02_degradation_conditioned_anchor_stream8.yaml"

echo "Submitting teacher-enabled stream8 speed-pivot VDA jobs from $CFG_DIR"

jid_q="$(submit_train "vdaT8Q" "$cfg_q")"
jid_d="$(submit_train "vdaT8D" "$cfg_d")"

bench_q_dir="${BENCH_ROOT}/quality_gate_anchor_stream8"
bench_d_dir="${BENCH_ROOT}/degradation_conditioned_anchor_stream8"
mkdir -p "$bench_q_dir" "$bench_d_dir"

jid_bq="$(
  submit_bench \
    "vdaBT8Q" \
    "$jid_q" \
    "runs/vda_teacher_stream8_pivot_20260401/01_quality_gate_anchor_stream8/config_snapshot.yaml" \
    "checkpoints/vda_teacher_stream8_pivot_20260401/01_quality_gate_anchor_stream8/best.pt" \
    "teacher_quality_gate_anchor_stream8" \
    "$bench_q_dir"
)"
jid_bd="$(
  submit_bench \
    "vdaBT8D" \
    "$jid_d" \
    "runs/vda_teacher_stream8_pivot_20260401/02_degradation_conditioned_anchor_stream8/config_snapshot.yaml" \
    "checkpoints/vda_teacher_stream8_pivot_20260401/02_degradation_conditioned_anchor_stream8/best.pt" \
    "teacher_degradation_conditioned_anchor_stream8" \
    "$bench_d_dir"
)"

echo "Submitted jobs:"
echo "  ${jid_q}   vdaT8Q   ${cfg_q}"
echo "  ${jid_d}   vdaT8D   ${cfg_d}"
echo "  ${jid_bq}  vdaBT8Q  afterok:${jid_q} quality-gate anchor stream8 benchmark"
echo "  ${jid_bd}  vdaBT8D  afterok:${jid_d} degradation-conditioned anchor stream8 benchmark"

echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
