#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) SLURM job script — VDA-small GT-only sanity training
# =============================================================================
#
# Purpose:
#   Train a small Video Depth Anything model (vendored architecture) with GT
#   supervision only. This is an architecture-vs-training-pipeline sanity test.
#
# Submit with:
#   sbatch jobs/run_train_vda_small_gt.sh
#
# Monitor with:
#   squeue --me
#   tail -f logs/vda_small_gt_<jobid>.out
# =============================================================================

#SBATCH --job-name=vda_small_gt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/vda_small_gt_%j.out
#SBATCH --error=./logs/vda_small_gt_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------------
PROJECT_DIR="/projects/b5dh/FocusMamba"
CONFIG="configs/experiments/tartanair_vda_small_gt.yaml"
VDA_CKPT="checkpoints/video_depth_anything_vits.pth"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "========================================"
echo " VDA-Small GT Training"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Config  : $CONFIG"
echo " Started : $(date)"
echo "========================================"

nvidia-smi --list-gpus

export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }

source .venv/bin/activate

which uv || { echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
echo "uv: $(uv --version)"

# ---------------------------------------------------------------------------
# Ensure small VDA checkpoint exists
# ---------------------------------------------------------------------------
if [ ! -f "$VDA_CKPT" ]; then
  echo "Checkpoint missing: $VDA_CKPT"
  echo "Downloading video_depth_anything_vits.pth from Hugging Face..."
  python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

Path("checkpoints").mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="depth-anything/Video-Depth-Anything-Small",
    filename="video_depth_anything_vits.pth",
    local_dir="checkpoints",
)
print(f"Downloaded checkpoint: {path}")
PY
fi

# ---------------------------------------------------------------------------
# Verify GPU visibility
# ---------------------------------------------------------------------------
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# Training — auto-resume from latest checkpoint
# ---------------------------------------------------------------------------
LATEST_CKPT="checkpoints/tartanair_vda_small_gt/latest.pt"
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
