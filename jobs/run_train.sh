#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) SLURM job script — FocusMamba full training
# Uses uv for dependency management.
# =============================================================================
#
# Before submitting:
#   1. Edit PROJECT_DIR and CONFIG below if needed.
#   2. Make sure uv is installed on Isambard:
#        curl -LsSf https://astral.sh/uv/install.sh | sh
#   3. Verify checkpoints are in ./checkpoints/ if using teachers:
#        checkpoints/da3_metric.safetensors
#        checkpoints/config.json               (DA3 config, same dir as above)
#        checkpoints/depth_pro.pt
#        checkpoints/metric_video_depth_anything_vitl.pth
#
# Submit with:
#   sbatch jobs/run_train.sh
#
# Monitor with:
#   squeue --me
#   tail -f logs/focusmamba_train_<jobid>.out
# =============================================================================

#SBATCH --job-name=focusmamba_train
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/focusmamba_train_%j.out
#SBATCH --error=./logs/focusmamba_train_%j.err

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
echo " FocusMamba Training"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Config  : $CONFIG"
echo " Started : $(date)"
echo "========================================"

nvidia-smi --list-gpus

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }

source .venv/bin/activate

which uv || { echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
echo "uv: $(uv --version)"

uv sync --locked

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Verify GPU visibility
# ---------------------------------------------------------------------------
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# Pre-flight: verify teacher imports (skip if no teachers in config)
# ---------------------------------------------------------------------------
echo "--- Checking teacher imports ---"
if ! python jobs/check_teachers.py --dry-run; then
    echo "WARNING: Teacher import check failed — continuing anyway (GT-only training)."
fi
echo "--- Pre-flight done ---"

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
    $RESUME_FLAG

echo "========================================"
echo " Finished : $(date)"
echo "========================================"
