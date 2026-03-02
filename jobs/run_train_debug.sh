#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) SLURM job script — FocusMamba SINGLE-GPU DEBUG
#
# Use this for quick iteration without consuming full-node resources.
# For 4-GPU distributed training, use jobs/run_train.sh instead.
# =============================================================================
#
# Submit with:
#   sbatch jobs/run_train_debug.sh
#
# Monitor with:
#   squeue --me
#   tail -f logs/focusmamba_debug_<jobid>.out
# =============================================================================

#SBATCH --job-name=focusmamba_debug
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/focusmamba_debug_%j.out
#SBATCH --error=./logs/focusmamba_debug_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# User config — edit these
# ---------------------------------------------------------------------------
PROJECT_DIR="/projects/b5dh/FocusMamba"
CONFIG="configs/experiments/tartanair_v2.yaml"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "========================================"
echo " FocusMamba Training (DEBUG / Single-GPU)"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Config  : $CONFIG"
echo " Started : $(date)"
echo "========================================"

nvidia-smi --list-gpus

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1        # dumps stack trace on segfault

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }

source .venv/bin/activate

which uv || { echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
echo "uv: $(uv --version)"
source .venv/bin/activate

# ---------------------------------------------------------------------------
# Verify GPU visibility
# ---------------------------------------------------------------------------
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# Training — auto-resume from latest checkpoint if it exists
# ---------------------------------------------------------------------------

# Derive checkpoint dir from config name (strip path and extension)
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
LATEST_CKPT="checkpoints/${CONFIG_NAME}/latest.pt"

RESUME_FLAG=""
if [ -f "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    RESUME_FLAG="--resume $LATEST_CKPT"
fi

python train.py \
    --config "$CONFIG" \
    --verbose \
    --debug \
    $RESUME_FLAG

echo "========================================"
echo " Finished : $(date)"
echo "========================================"
