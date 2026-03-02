#!/bin/bash
# =============================================================================
# Smoke test — run teacher caching on a handful of clips and save outputs
# so they can be manually inspected in zz_dataset.ipynb before committing
# to an overnight run on the full 765 GB dataset.
#
# Runs directly (no SLURM needed):
#   bash jobs/smoke_test_cache.sh
#
# Or on the cluster interactively:
#   srun --partition=workq --gres=gpu:1 --mem=40G --cpus-per-task=4 \
#        --pty bash jobs/smoke_test_cache.sh
#
# After it completes, open zz_dataset.ipynb and run the
# "Smoke-test: inspect cached teacher outputs" cells.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — edit ENV and N_CLIPS as needed
# ---------------------------------------------------------------------------
PROJECT_DIR="/projects/b5dh/FocusMamba"
CONFIG="configs/experiments/tartanair_v2.yaml"
ENV="Slaughter"           # which environment to test
N_CLIPS=5                        # how many clips to cache (keeps it fast)
SMOKE_CACHE_DIR="/projects/b5dh/data/teacher_cache/smoke_test"

echo "========================================"
echo " FocusMamba — Teacher Cache Smoke Test"
echo " Env       : $ENV"
echo " N clips   : $N_CLIPS"
echo " Cache dir : $SMOKE_CACHE_DIR"
echo " Started   : $(date)"
echo "========================================"

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Activate environment
source .venv/bin/activate

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# Dry run — confirms teachers and dataset load cleanly before spending GPU time
# ---------------------------------------------------------------------------
echo ""
echo "--- Dry run ---"
python tools/cache_teacher_labels.py \
    --config "$CONFIG" \
    --cache-dir "$SMOKE_CACHE_DIR" \
    --envs "$ENV" \
    --max-clips "$N_CLIPS" \
    --split train \
    --dry-run
echo "--- Dry run OK ---"
echo ""

# ---------------------------------------------------------------------------
# Cache N_CLIPS clips from the train split of ENV
# ---------------------------------------------------------------------------
echo "--- Caching $N_CLIPS clips from $ENV (split=train) ---"
python tools/cache_teacher_labels.py \
    --config "$CONFIG" \
    --cache-dir "$SMOKE_CACHE_DIR" \
    --envs "$ENV" \
    --max-clips "$N_CLIPS" \
    --split train \
    --num-workers 2 \
    --batch-size 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
N_FILES=$(find "$SMOKE_CACHE_DIR" -name "*.npy" 2>/dev/null | wc -l || echo 0)
DISK=$(du -sh "$SMOKE_CACHE_DIR" 2>/dev/null | cut -f1 || echo "unknown")

echo "========================================"
echo " Smoke test complete."
echo " Cache dir  : $SMOKE_CACHE_DIR"
echo " .npy files : $N_FILES  (expect $((N_CLIPS * 3)) = $N_CLIPS clips × 3 teachers)"
echo " Disk used  : $DISK"
echo " Finished   : $(date)"
echo "========================================"
echo ""
echo "Next: open zz_dataset.ipynb and run the 'Smoke-test' cells."
echo "  SMOKE_CACHE_DIR = $SMOKE_CACHE_DIR"
echo "  DATA_ROOT       = $(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['data']['root'])")"
