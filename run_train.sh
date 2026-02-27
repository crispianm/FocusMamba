#!/usr/bin/env bash
# ============================================================================
# run_train.sh — Submit FocusMamba training to an HPC SLURM cluster
# ============================================================================
#
# Usage:
#   sbatch run_train.sh [CONFIG] [RESUME_CHECKPOINT]
#
# Examples:
#   sbatch run_train.sh
#   sbatch run_train.sh configs/experiments/distill_v1.yaml
#   sbatch run_train.sh configs/base.yaml checkpoints/mamba_depth/latest.pt
#
# The script activates the project's uv virtual environment at /.venv and
# runs train.py.  Override CONFIG and RESUME_CHECKPOINT via positional args
# or by editing the defaults below.
# ============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=focusmamba_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# ── Defaults (override via positional args) ───────────────────────────────────
CONFIG="${1:-configs/base.yaml}"
RESUME="${2:-}"

# ── Environment setup ─────────────────────────────────────────────────────────
set -euo pipefail

# Project root is the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the uv virtual environment
VENV="${SCRIPT_DIR}/.venv"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    # Fall back to the system-wide location used on some HPC clusters
    VENV="/.venv"
fi
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
else
    echo "WARNING: virtual environment not found at ${SCRIPT_DIR}/.venv or /.venv"
    echo "Attempting to use system Python; install deps with: uv sync"
fi

echo "============================================================"
echo "  FocusMamba Training"
echo "  Config  : ${CONFIG}"
echo "  Resume  : ${RESUME:-<none>}"
echo "  Python  : $(python --version 2>&1)"
echo "  Node    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'n/a')"
echo "  Time    : $(date)"
echo "============================================================"

# ── Launch training ───────────────────────────────────────────────────────────
ARGS=(--config "${CONFIG}")
if [[ -n "${RESUME}" ]]; then
    ARGS+=(--resume "${RESUME}")
fi

python train.py "${ARGS[@]}"

echo "============================================================"
echo "  Training finished at $(date)"
echo "============================================================"
