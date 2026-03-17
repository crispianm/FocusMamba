#!/bin/bash
# =============================================================================
# SLURM validation sweep for pretrained Video Depth Anything checkpoints.
# Defaults are aligned to the degraded validation setup used by
# focused_redirection_20260310/05_vdal_degraded_stream, while using the
# official VDA inference recipe (offline, fp16, 518x518 teacher-style input).
# =============================================================================

#SBATCH --job-name=vda_val
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
CONFIG="${CONFIG:-configs/experiments/generated/focused_redirection_20260310/05_vdal_degraded_stream.yaml}"
SMALL_CHECKPOINT="${SMALL_CHECKPOINT:-checkpoints/metric_video_depth_anything_vits.pth}"
LARGE_CHECKPOINT="${LARGE_CHECKPOINT:-checkpoints/metric_video_depth_anything_vitl.pth}"
MODELS="${MODELS:-small,large}"
NUM_CLIPS="${NUM_CLIPS:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODE="${MODE:-offline}"
AMP="${AMP:-fp16}"
INFERENCE_SIZE="${INFERENCE_SIZE:-518,518}"
STRICT_CHECKPOINT="${STRICT_CHECKPOINT:-1}"
DEGRADATION_LEVELS="${DEGRADATION_LEVELS:-vdal_degraded,10,1.0}"
SKIP_CLEAN="${SKIP_CLEAN:-1}"
CLEAN_ONLY="${CLEAN_ONLY:-0}"
RESUME="${RESUME:-0}"
LOG_EVERY="${LOG_EVERY:-50}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODELS//,/+}"
INFER_TAG="${INFERENCE_SIZE//,/x}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/vda_validation_baselines/${STAMP}_${MODEL_TAG}_${MODE}_${INFER_TAG}}"

echo "========================================"
echo " VDA Validation Baselines"
echo " Job ID   : ${SLURM_JOB_ID:-local}"
echo " Node     : $(hostname)"
echo " Config   : ${CONFIG}"
echo " Models   : ${MODELS}"
echo " InferSz  : ${INFERENCE_SIZE}"
echo " Output   : ${OUTPUT_DIR}"
echo " Started  : $(date -u)"
echo "========================================"

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: Config not found: $CONFIG"
  exit 1
fi
if [ ! -f "$SMALL_CHECKPOINT" ]; then
  echo "ERROR: Small VDA checkpoint not found: $SMALL_CHECKPOINT"
  exit 1
fi
if [ ! -f "$LARGE_CHECKPOINT" ]; then
  echo "ERROR: Large VDA checkpoint not found: $LARGE_CHECKPOINT"
  exit 1
fi

source .venv/bin/activate

nvidia-smi --list-gpus || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

LEVEL_ARGS=()
IFS=';' read -r -a LEVEL_SPECS <<< "$DEGRADATION_LEVELS"
for level_spec in "${LEVEL_SPECS[@]}"; do
  if [ -n "$level_spec" ]; then
    LEVEL_ARGS+=(--degradation-level "$level_spec")
  fi
done

EXTRA_ARGS=()
if [ "$SKIP_CLEAN" = "1" ]; then
  EXTRA_ARGS+=(--skip-clean)
fi
if [ "$CLEAN_ONLY" = "1" ]; then
  EXTRA_ARGS+=(--clean-only)
fi
if [ "$STRICT_CHECKPOINT" = "1" ]; then
  EXTRA_ARGS+=(--strict-checkpoint)
fi
if [ "$RESUME" = "1" ]; then
  EXTRA_ARGS+=(--resume)
fi

set -x
python tools/eval_vda_validation_sweep.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --small-checkpoint "$SMALL_CHECKPOINT" \
  --large-checkpoint "$LARGE_CHECKPOINT" \
  --models "$MODELS" \
  --num-clips "$NUM_CLIPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --inference-size "$INFERENCE_SIZE" \
  --mode "$MODE" \
  --amp "$AMP" \
  --log-every "$LOG_EVERY" \
  "${EXTRA_ARGS[@]}" \
  "${LEVEL_ARGS[@]}"
set +x

echo "========================================"
echo " Finished : $(date -u)"
echo " Results  : ${OUTPUT_DIR}/summary.json"
echo "            ${OUTPUT_DIR}/summary.csv"
echo "========================================"
