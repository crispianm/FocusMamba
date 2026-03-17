#!/bin/bash
# =============================================================================
# Upstream Video Depth Anything streaming benchmark on degraded TartanAir val.
# Uses the original repo's streaming inference path against a deterministic
# degraded export of the current TartanAir validation split.
# =============================================================================

#SBATCH --job-name=vda_up_bench
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: Submit this script with sbatch so it runs on a compute node."
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
REPO_ROOT="${REPO_ROOT:-/projects/b5dh/repos/Video-Depth-Anything}"
CONFIG="${CONFIG:-configs/experiments/generated/focused_redirection_20260310/05_vdal_degraded_stream.yaml}"
MODELS="${MODELS:-vits,vitl}"
STREAM_SCOPE="${STREAM_SCOPE:-both}"
METRIC_SUITE="${METRIC_SUITE:-both}"
INPUT_SIZE="${INPUT_SIZE:-518}"
DEVICE="${DEVICE:-auto}"
FP32="${FP32:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE_EXPORT="${OVERWRITE_EXPORT:-0}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/vda_upstream_validation_benchmark/${STAMP}}"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/.runtime_cache}"
DEBUG_MAX_TRAJECTORIES="${DEBUG_MAX_TRAJECTORIES:-}"
DEBUG_MAX_GROUPS_PER_MODE="${DEBUG_MAX_GROUPS_PER_MODE:-}"

echo "========================================"
echo " Upstream VDA TartanAir Benchmark"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : $(hostname)"
echo " Config   : ${CONFIG}"
echo " Repo     : ${REPO_ROOT}"
echo " Models   : ${MODELS}"
echo " Scope    : ${STREAM_SCOPE}"
echo " Metrics  : ${METRIC_SUITE}"
echo " Output   : ${OUTPUT_ROOT}"
echo " Started  : $(date -u)"
echo "========================================"

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs "$OUTPUT_ROOT"
mkdir -p "$CACHE_ROOT"/xdg "$CACHE_ROOT"/torchinductor "$CACHE_ROOT"/triton "$CACHE_ROOT"/torch_home
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch_home}"

if [ ! -d "$REPO_ROOT" ]; then
  echo "ERROR: Upstream repo not found: $REPO_ROOT"
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: Config not found: $CONFIG"
  exit 1
fi

source .venv/bin/activate

nvidia-smi --list-gpus || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

ARGS=(
  python tools/benchmark_vda_upstream_tartanair.py
  --config "$CONFIG"
  --repo-root "$REPO_ROOT"
  --output-root "$OUTPUT_ROOT"
  --models "$MODELS"
  --stream-scope "$STREAM_SCOPE"
  --metric-suite "$METRIC_SUITE"
  --input-size "$INPUT_SIZE"
  --device "$DEVICE"
)

if [ "$FP32" = "1" ]; then
  ARGS+=(--fp32)
fi
if [ "$SKIP_EXISTING" = "1" ]; then
  ARGS+=(--skip-existing)
fi
if [ "$OVERWRITE_EXPORT" = "1" ]; then
  ARGS+=(--overwrite-export)
fi
if [ -n "$DEBUG_MAX_TRAJECTORIES" ]; then
  ARGS+=(--debug-max-trajectories "$DEBUG_MAX_TRAJECTORIES")
fi
if [ -n "$DEBUG_MAX_GROUPS_PER_MODE" ]; then
  ARGS+=(--debug-max-groups-per-mode "$DEBUG_MAX_GROUPS_PER_MODE")
fi

set -x
"${ARGS[@]}"
set +x

echo "========================================"
echo " Finished : $(date -u)"
echo " Results  : ${OUTPUT_ROOT}/summary"
echo " Export   : ${OUTPUT_ROOT}/dataset_export"
echo "========================================"
