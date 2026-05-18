#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) SLURM job — Sanity check: clean fine-tune of VDA-Small
#
# Trains VDA-Small from pretrained weights on 32 clean TartanAir v2 trajectories
# for 10 epochs. Pass criterion: val abs_rel decreases epoch-over-epoch.
#
# Submit: sbatch jobs/run_sanity_clean.sh
# Monitor: tail -f logs/sanity_clean_<jobid>.out
# =============================================================================

#SBATCH --job-name=sanity_clean
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/sanity_clean_%j.out
#SBATCH --error=./logs/sanity_clean_%j.err

set -euo pipefail

PROJECT_DIR="/projects/b5dh/FocusMamba"
CONFIG="configs/experiments/sanity_check_clean.yaml"

echo "========================================"
echo " Sanity Check: Clean Fine-tune"
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

TORCHRUN_HOME_ROOT="${TORCHRUN_HOME_ROOT:-${HOME}/.focusmamba/isambard}"
TORCHRUN_TMP_ROOT="${TORCHRUN_HOME_ROOT}/tmp/${SLURM_JOB_ID:-local}"
TORCHRUN_LOG_ROOT="${TORCHRUN_HOME_ROOT}/torchrun_logs/${SLURM_JOB_ID:-local}"
mkdir -p "$TORCHRUN_TMP_ROOT" "$TORCHRUN_LOG_ROOT"
export TMPDIR="$TORCHRUN_TMP_ROOT"
export TEMP="$TORCHRUN_TMP_ROOT"
export TMP="$TORCHRUN_TMP_ROOT"

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }
source .venv/bin/activate

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

MASTER_PORT="${MASTER_PORT:-$((15000 + (${SLURM_JOB_ID:-0} % 40000)))}"
echo "MASTER_PORT=$MASTER_PORT"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${MASTER_PORT}" \
    --log-dir "$TORCHRUN_LOG_ROOT" \
    train.py \
    --config "$CONFIG" \
    --verbose

echo "========================================"
echo " Finished : $(date)"
echo "========================================"
