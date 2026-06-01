#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) — degraded-erosion investigation, ONE config on 1 GPU.
# Trains a short small-data run (saves a per-epoch checkpoint ladder) then runs
# the severity x seed sweep on that ladder for both VKITTI and TartanAir.
#
# Submit:  sbatch jobs/erosion_run.sh <config_name>      # e.g. base_small, c1_no_ema
# =============================================================================

#SBATCH --job-name=erosion
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/erosion_%x_%j.out
#SBATCH --error=./logs/erosion_%x_%j.err

set -euo pipefail

NAME="${1:?usage: sbatch jobs/erosion_run.sh <config_name>}"
PROJECT_DIR="/projects/b5dh/FocusMamba"
INV="degraded_erosion_investigation"
CONFIG="${INV}/configs/${NAME}.yaml"
CKPT_DIR="/projects/b5dh/${INV}/ckpts/${NAME}"
BASELINE="checkpoints/metric_video_depth_anything_vits.pth"

echo "==== erosion run: ${NAME} | job ${SLURM_JOB_ID} | $(hostname) | $(date) ===="
nvidia-smi --list-gpus

export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1

TORCHRUN_HOME_ROOT="${TORCHRUN_HOME_ROOT:-${HOME}/.focusmamba/isambard}"
TORCHRUN_TMP_ROOT="${TORCHRUN_HOME_ROOT}/tmp/${SLURM_JOB_ID:-local}"
TORCHRUN_LOG_ROOT="${TORCHRUN_HOME_ROOT}/torchrun_logs/${SLURM_JOB_ID:-local}"
mkdir -p "$TORCHRUN_TMP_ROOT" "$TORCHRUN_LOG_ROOT"
export TMPDIR="$TORCHRUN_TMP_ROOT" TEMP="$TORCHRUN_TMP_ROOT" TMP="$TORCHRUN_TMP_ROOT"

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd $PROJECT_DIR"; exit 1; }
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MASTER_PORT="${MASTER_PORT:-$((15000 + (${SLURM_JOB_ID:-0} % 40000)))}"

# ---- train (1 GPU, per-epoch ladder via save_every_n_epochs in the config) ----
torchrun --nproc_per_node=1 --nnodes=1 --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${MASTER_PORT}" --log-dir "$TORCHRUN_LOG_ROOT" \
    train.py --config "$CONFIG" --verbose

echo "==== training done: $(date); running severity sweeps ===="

# ---- Phase B: severity x seed sweep on the per-epoch ladder (eval only) ----
for DS in vkitti tartanair; do
    python ${INV}/scripts/eval_severity_sweep.py \
        --config "$CONFIG" \
        --checkpoint-dir "$CKPT_DIR" \
        --baseline-checkpoint "$BASELINE" \
        --dataset "$DS" \
        --severities 0.0,0.25,0.5,0.75,1.0 \
        --seeds 1,2,3 \
        --max-clips 40 --batch-size 2 \
        --out "${INV}/runs/${NAME}/sweep_${DS}.csv" \
        || echo "WARN: sweep ${DS} failed"
done

echo "==== ALL DONE ${NAME}: $(date) ===="
