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
#   3. Run teacher label caching BEFORE this job (one-time, ~2-4 h):
#        sbatch jobs/run_cache_teachers.sh
#      This eliminates ~2.5 s/it of teacher inference overhead and is the
#      single largest training speed-up.  Using the cache the expected
#      throughput on a GH200 is ~0.25-0.35 s/it (vs 2.68 s/it uncached).
#   4. Verify checkpoints are in ./checkpoints/ if using teachers:
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
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/focusmamba_train_%j.out
#SBATCH --error=./logs/focusmamba_train_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# User config — edit these
# ---------------------------------------------------------------------------
PROJECT_DIR="/projects/b5dh/FocusMamba"
CONFIG="configs/experiments/tartanair_GT.yaml"

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

# GH200 NVLink C2C — force NCCL to use NVLink for intra-node comms
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
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

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    train.py \
    --config "$CONFIG" \
    --verbose \
    --debug \
    $RESUME_FLAG

echo "========================================"
echo " Finished : $(date)"
echo "========================================"
