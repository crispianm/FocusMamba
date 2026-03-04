#!/bin/bash
# =============================================================================
# Isambard-AI (H100/GH200) SLURM job script
# VDA-small (scratch) + GT + 3-teacher distillation
#
# Default usage:
#   sbatch jobs/run_train_vda_small_scratch_distill.sh
#
# Override config at submit-time (used by ablation launcher):
#   sbatch --export=ALL,CONFIG=configs/experiments/generated/<run>/<exp>.yaml \
#          jobs/run_train_vda_small_scratch_distill.sh
# =============================================================================

#SBATCH --job-name=vda_small_scratch
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
CONFIG="${CONFIG:-configs/experiments/tartanair_vda_small_scratch_distill_full.yaml}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "========================================"
echo " VDA-Small Scratch Distillation Training"
echo " Job ID   : ${SLURM_JOB_ID:-local}"
echo " Node     : $(hostname)"
echo " Config   : $CONFIG"
echo " Started  : $(date -u)"
echo "========================================"

nvidia-smi --list-gpus || true

export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONFAULTHANDLER=1

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: Config not found: $CONFIG"
  exit 1
fi

source .venv/bin/activate

which uv >/dev/null || {
  echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}
echo "uv: $(uv --version)"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# Read checkpoint/cache settings from YAML
# ---------------------------------------------------------------------------
readarray -t CFG_META < <(python - <<'PY' "$CONFIG"
import sys
from pathlib import Path
import yaml

p = Path(sys.argv[1])
with p.open("r") as f:
    cfg = yaml.safe_load(f) or {}

training = cfg.get("training", {}) or {}
distill = cfg.get("distillation", {}) or {}
teachers = cfg.get("teachers", []) or []
data = cfg.get("data", {}) or {}

checkpoint_dir = training.get("checkpoint_dir", f"checkpoints/{p.stem}")
skip_live = bool(training.get("skip_live_teachers", bool(data.get("teacher_cache_dir"))))
distill_enabled = bool(distill.get("enabled", False))
cache_dir = data.get("teacher_cache_dir", "") or ""
teacher_count = len([t for t in teachers if t.get("enabled", True)])

print(checkpoint_dir)
print("1" if skip_live else "0")
print("1" if distill_enabled else "0")
print(cache_dir)
print(str(teacher_count))
PY
)

CHECKPOINT_DIR="${CFG_META[0]}"
SKIP_LIVE="${CFG_META[1]}"
DISTILL_ENABLED="${CFG_META[2]}"
CACHE_DIR="${CFG_META[3]}"
TEACHER_COUNT="${CFG_META[4]}"

# Distillation guard: fail fast if config asks for cache-only teachers but cache is unavailable.
if [ "$DISTILL_ENABLED" = "1" ] && [ "$TEACHER_COUNT" -gt 0 ] && [ "$SKIP_LIVE" = "1" ]; then
  if [ -z "$CACHE_DIR" ] || [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: distillation is enabled with skip_live_teachers=true, but teacher_cache_dir is missing."
    echo "       CONFIG=$CONFIG"
    exit 1
  fi
  if ! find "$CACHE_DIR" -name '*.npy' -print -quit | grep -q .; then
    echo "ERROR: teacher_cache_dir exists but contains no .npy cached labels: $CACHE_DIR"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Auto-resume from latest checkpoint if present
# ---------------------------------------------------------------------------
LATEST_CKPT="${CHECKPOINT_DIR}/latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST_CKPT" ]; then
  echo "Resuming from $LATEST_CKPT"
  RESUME_FLAG="--resume $LATEST_CKPT"
fi

# Avoid DDP rendezvous port collisions when many jobs share a node.
MASTER_PORT="${MASTER_PORT:-$((15000 + (${SLURM_JOB_ID:-0} % 40000)))}"
echo "MASTER_PORT=$MASTER_PORT"

set -x
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="localhost:${MASTER_PORT}" \
  train.py \
  --config "$CONFIG" \
  --verbose \
  --debug \
  $RESUME_FLAG \
  $EXTRA_TRAIN_ARGS
set +x

echo "========================================"
echo " Finished : $(date -u)"
echo "========================================"
