#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) SLURM job — Degraded fine-tune of VDA-Small at data scale.
#
# Fine-tunes the metric VDA-S checkpoint on FULL 74-env TartanAir-v2 + VKITTI with
# v2_lowlight degradation (SSI+TGM, 518), then evaluates clean-KITTI transfer.
# Confirms the local result holds at scale and that multi-GPU/HPC does not drop
# performance, before any model-architecture changes.
#
# Submit:  sbatch jobs/run_degraded_ft_vkitti_hpc.sh
# Monitor: tail -f logs/degraded_ft_vkitti_<jobid>.out
# =============================================================================

#SBATCH --job-name=degraded_ft_vkitti
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/degraded_ft_vkitti_%j.out
#SBATCH --error=./logs/degraded_ft_vkitti_%j.err

set -euo pipefail

PROJECT_DIR="/projects/b5dh/FocusMamba"
CONFIG="configs/experiments/degraded_ft_vkitti_tartanair_hpc.yaml"

echo "========================================"
echo " Degraded FT (VKITTI + full TartanAir)"
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
echo " Training finished : $(date)"
echo "========================================"

# --- Clean-KITTI transfer eval of the degraded-FT model (single GPU) ---------
BEST="checkpoints/local/degraded_ft_vkitti_hpc/best.pt"
if [ -f "$BEST" ]; then
    echo " KITTI eval of $BEST : $(date)"
    python benchmark_kitti_vda.py \
        --checkpoints "$BEST" \
        --is-metric true \
        --output-dir runs/kitti_eval/degraded_ft_vkitti_hpc \
        || echo "WARN: KITTI eval failed"
    echo " KITTI eval done : $(date)"
else
    echo " WARN: $BEST not found; skipping KITTI eval"
fi

echo "========================================"
echo " ALL DONE : $(date)"
echo "========================================"
