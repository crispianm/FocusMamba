#!/bin/bash
# =============================================================================
# SLURM job script for matched checkpoint evaluation on the TartanAir val split.
# =============================================================================

#SBATCH --job-name=vda_rescheck
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
CONFIG="${CONFIG:-}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AMP="${AMP:-auto}"
INPUT_SOURCE="${INPUT_SOURCE:-auto}"
MODE_OVERRIDE="${MODE_OVERRIDE:-}"
INFERENCE_SIZE="${INFERENCE_SIZE:-}"
CLIP_LIMIT="${CLIP_LIMIT:-}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
STRICT_CHECKPOINT="${STRICT_CHECKPOINT:-1}"
RESPECT_CONFIG_VAL_BUDGET="${RESPECT_CONFIG_VAL_BUDGET:-0}"
LOG_EVERY="${LOG_EVERY:-50}"
FLUSH_EVERY="${FLUSH_EVERY:-25}"
REFERENCE_PER_CLIP_CSV="${REFERENCE_PER_CLIP_CSV:-}"
RESUME="${RESUME:-0}"
CLIP_SHARD_INDEX="${CLIP_SHARD_INDEX:-0}"
CLIP_SHARD_COUNT="${CLIP_SHARD_COUNT:-1}"

echo "========================================"
echo " Checkpoint Evaluation"
echo " Job ID     : ${SLURM_JOB_ID:-local}"
echo " Node       : $(hostname)"
echo " Config     : ${CONFIG}"
echo " Checkpoint : ${CHECKPOINT}"
echo " Output     : ${OUTPUT_DIR}"
echo " Started    : $(date -u)"
echo "========================================"

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs

if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
  echo "ERROR: CONFIG is missing or not a file: $CONFIG"
  exit 1
fi
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: CHECKPOINT is missing or not a file: $CHECKPOINT"
  exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
  echo "ERROR: OUTPUT_DIR must be set"
  exit 1
fi

source .venv/bin/activate

nvidia-smi --list-gpus || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

EXTRA_ARGS=()
if [ "$STRICT_CHECKPOINT" = "1" ]; then
  EXTRA_ARGS+=(--strict-checkpoint)
fi
if [ "$RESPECT_CONFIG_VAL_BUDGET" = "1" ]; then
  EXTRA_ARGS+=(--respect-config-val-budget)
fi
if [ -n "$MODE_OVERRIDE" ]; then
  EXTRA_ARGS+=(--mode "$MODE_OVERRIDE")
fi
if [ -n "$INFERENCE_SIZE" ]; then
  EXTRA_ARGS+=(--inference-size "$INFERENCE_SIZE")
fi
if [ -n "$CLIP_LIMIT" ]; then
  EXTRA_ARGS+=(--clip-limit "$CLIP_LIMIT")
fi
if [ -n "$REFERENCE_PER_CLIP_CSV" ]; then
  EXTRA_ARGS+=(--reference-per-clip-csv "$REFERENCE_PER_CLIP_CSV")
fi
if [ "$RESUME" = "1" ]; then
  EXTRA_ARGS+=(--resume)
fi

set -x
python tools/evaluate_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --amp "$AMP" \
  --input-source "$INPUT_SOURCE" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --log-every "$LOG_EVERY" \
  --flush-every "$FLUSH_EVERY" \
  --clip-shard-index "$CLIP_SHARD_INDEX" \
  --clip-shard-count "$CLIP_SHARD_COUNT" \
  "${EXTRA_ARGS[@]}"
set +x

echo "========================================"
echo " Finished : $(date -u)"
echo "========================================"
