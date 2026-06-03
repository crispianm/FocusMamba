#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) — clean<->degraded consistency prototype, ONE arm on 1 GPU.
# Trains a 2-env-TartanAir + VKITTI testbed arm (WDDA, 5e-5, 6 epochs, ladder)
# then runs the severity x seed sweep on its ladder. Single-GPU to match the
# local control's optimisation exactly (batch 1 x grad_accum 4).
#
# Submit:  sbatch jobs/cc_run.sh <arm>      # e.g. si_w1p0, logl1_w0p5, c0_control
# =============================================================================

#SBATCH --job-name=cc_arm
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=18
#SBATCH --output=./logs/cc_%x_%j.out
#SBATCH --error=./logs/cc_%x_%j.err

set -u
NAME="${1:?usage: sbatch jobs/cc_run.sh <arm>}"
PROJECT_DIR="/projects/b5dh/FocusMamba"
P="clean_consistency_prototype"
CFG="$P/configs_hpc/${NAME}.yaml"
CKPT="/projects/b5dh/$P/ckpts/${NAME}"
SWEEP="degraded_erosion_investigation/scripts/eval_severity_sweep.py"

echo "==== cc arm ${NAME} | job ${SLURM_JOB_ID} | $(hostname) | $(date) ===="
nvidia-smi --list-gpus

export NCCL_P2P_LEVEL=NVL NCCL_IB_DISABLE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
TR="${HOME}/.focusmamba/isambard/tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$TR"
export TMPDIR="$TR" TEMP="$TR" TMP="$TR"

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd $PROJECT_DIR"; exit 1; }
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# single-GPU (no torchrun) -> identical optimisation to the local control
CUDA_VISIBLE_DEVICES=0 python train.py --config "$CFG" --verbose
TRAIN_RC=$?
echo "==== training ${NAME} done rc=$TRAIN_RC $(date) ===="
if [ "$TRAIN_RC" -ne 0 ]; then echo "train.py rc=$TRAIN_RC; skipping sweep"; exit "$TRAIN_RC"; fi

for DS in vkitti tartanair; do
    CUDA_VISIBLE_DEVICES=0 python "$SWEEP" \
        --config "$CFG" --checkpoint-dir "$CKPT" \
        --baseline-checkpoint checkpoints/metric_video_depth_anything_vits.pth \
        --dataset "$DS" --severities 0.0,0.25,0.5,0.75,1.0 --seeds 1,2,3 \
        --max-clips 40 --batch-size 2 \
        --out "$P/runs/${NAME}/sweep_${DS}.csv" || echo "WARN sweep ${DS} failed"
done
echo "==== ALL DONE ${NAME} $(date) ===="
