#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) SLURM job script — FocusMamba trial training
# Uses uv for dependency management.
# =============================================================================
#
# Before submitting:
#   1. Edit PROJECT_DIR below.
#   2. Make sure uv is installed on Isambard:
#        curl -LsSf https://astral.sh/uv/install.sh | sh
#   3. Verify checkpoints are in ./checkpoints/:
#        checkpoints/da3_metric.safetensors
#        checkpoints/config.json               (DA3 config, same dir as above)
#        checkpoints/depth_pro.pt
#        checkpoints/metric_video_depth_anything_vitl.pth
#   4. Run the pre-flight check first (optional but recommended):
#        uv run python jobs/check_teachers.py
#
# Submit with:
#   sbatch jobs/run_test.sh
#
# Monitor with:
#   squeue --me
#   tail -f focusmamba_trial_<jobid>.out
# =============================================================================

#SBATCH --job-name=focusmamba_trial
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00      # 24-hour limit; requeue with --dependency=singleton
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/focusmamba_trial_%j.out
#SBATCH --error=./logs/focusmamba_trial_%j.err

set -euo pipefail



# ---------------------------------------------------------------------------
# User config — edit these
# ---------------------------------------------------------------------------
PROJECT_DIR="/projects/b5dh/FocusMamba"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "========================================"
echo " FocusMamba Trial Training"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Started : $(date)"
echo "========================================"

nvidia-smi --list-gpus

source /projects/b5dh/FocusMamba/.venv/bin/activate

which uv || { echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
echo "uv: $(uv --version)"

# Move into project directory
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }

# Sync all dependencies from uv.lock into .venv (creates .venv if missing)
uv sync --locked

# Install ml-depth-pro (not in pyproject.toml — teacher repo has its own setup)
uv pip install -e teachers/ml-depth-pro --quiet

# Activate the venv so plain `python` calls use it for the rest of the script
source .venv/bin/activate

# ---------------------------------------------------------------------------
# Verify GPU visibility
# ---------------------------------------------------------------------------
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# Pre-flight: verify teacher imports
# ---------------------------------------------------------------------------
echo "--- Checking teacher imports ---"
if ! python jobs/check_teachers.py --dry-run; then
    echo "ERROR: Teacher import check failed. Aborting."
    exit 1
fi
echo "--- Teacher imports OK ---"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
CONFIG="configs/experiments/trial_youtube_vos.yaml"

# Check if a latest checkpoint exists for resuming
RESUME_FLAG=""
LATEST_CKPT="checkpoints/trial_youtube_vos/latest.pt"
if [ -f "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    RESUME_FLAG="--resume $LATEST_CKPT"
fi

python test_training.py \
    --config "$CONFIG" \
    $RESUME_FLAG

echo "========================================"
echo " Finished : $(date)"
echo "========================================"
