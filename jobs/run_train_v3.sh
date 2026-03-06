#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) SLURM job script for the current tartanair_v3 config.
# =============================================================================

#SBATCH --job-name=focusmamba_v3
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/focusmamba_v3_%j.out
#SBATCH --error=./logs/focusmamba_v3_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
CONFIG="${CONFIG:-configs/experiments/tartanair_v3.yaml}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

echo "========================================"
echo " FocusMamba v3 Training"
echo " Job ID   : ${SLURM_JOB_ID:-local}"
echo " Node     : $(hostname)"
echo " Config   : ${CONFIG}"
echo " Started  : $(date -u)"
echo "========================================"

nvidia-smi --list-gpus || true

export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONFAULTHANDLER=1

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }

if [[ ! -f "$CONFIG" ]]; then
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

CHECKPOINT_DIR="$(python - <<'PY' "$CONFIG"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r") as f:
    cfg = yaml.safe_load(f) or {}

training = cfg.get("training", {}) or {}
checkpoint_dir = training.get("checkpoint_dir", f"checkpoints/{config_path.stem}")
print(checkpoint_dir)
PY
)"
LATEST_CKPT="${CHECKPOINT_DIR}/latest.pt"

RESUME_FLAG=""
if [[ -f "$LATEST_CKPT" ]]; then
  echo "Resuming from $LATEST_CKPT"
  RESUME_FLAG="--resume $LATEST_CKPT"
fi

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
