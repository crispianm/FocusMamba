#!/bin/bash
# =============================================================================
# Isambard-AI (GH200) SLURM array job — Pre-cache teacher pseudo-labels
# One array task per TartanAir v2 environment (~11 min per env on a GH200).
# =============================================================================
#
# SUBMIT (all 76 environments in parallel)
#   sbatch jobs/run_cache_teachers.sh
#
# SUBMIT (specific tasks only, e.g. first 10)
#   sbatch --array=0-9 jobs/run_cache_teachers.sh
#
# RESUBMIT failed tasks (check IDs with: sacct -j <jobid> --format=JobID,State)
#   sbatch --array=<failed_ids> jobs/run_cache_teachers.sh
#
# MONITOR
#   squeue --me
#   tail -f logs/focusmamba_cache_<jobid>_<taskid>.out
#
# Once all tasks finish, submit training:
#   sbatch jobs/run_train.sh
# =============================================================================

#SBATCH --job-name=fm_cache
#SBATCH --partition=workq
#SBATCH --array=0-73
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/focusmamba_cache_%A_%a.out
#SBATCH --error=./logs/focusmamba_cache_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment list — alphabetical, 76 entries matching the dataset on disk.
# Each array task picks one environment by its SLURM_ARRAY_TASK_ID index.
# ---------------------------------------------------------------------------
ENVS=(
    AbandonedCable
    AbandonedFactory
    AbandonedFactory2
    AbandonedSchool
    AmericanDiner
    AmusementPark
    AncientTowns
    Antiquity3D
    Apocalyptic
    ArchVizTinyHouseDay
    ArchVizTinyHouseNight
    BrushifyMoon
    CarWelding
    CastleFortress
    CoalMine
    ConstructionSite
    CountryHouse
    Cyberpunk
    CyberPunkDowntown
    DesertGasStation
    Downtown
    EndofTheWorld
    FactoryWeather
    Fantasy
    ForestEnv
    Gascola
    GothicIsland
    GreatMarsh
    HongKong
    Hospital
    House
    HQWesternSaloon
    IndustrialHangar
    JapaneseAlley
    JapaneseCity
    MiddleEast
    ModernCityDowntown
    ModularNeighborhood
    ModularNeighborhoodIntExt
    ModUrbanCity
    NordicHarbor
    Ocean
    Office
    OldBrickHouseDay
    OldBrickHouseNight
    OldIndustrialCity
    OldScandinavia
    OldTownFall
    OldTownNight
    OldTownSummer
    OldTownWinter
    PolarSciFi
    Prison
    Restaurant
    RetroOffice
    Rome
    Ruins
    SeasideTown
    SeasonalForestAutumn
    SeasonalForestSpring
    SeasonalForestSummerNight
    SeasonalForestWinter
    SeasonalForestWinterNight
    Sewerage
    ShoreCaves
    Slaughter
    SoulCity
    Supermarket
    TerrainBlending
    UrbanConstruction
    VictorianStreet
    WaterMillDay
    WaterMillNight
    WesternDesertTown
)

ENV="${ENVS[$SLURM_ARRAY_TASK_ID]}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR="/projects/b5dh/FocusMamba"
CONFIG="configs/experiments/tartanair_v2.yaml"
CACHE_DIR="/projects/b5dh/data/teacher_cache/tartanair_v2"

echo "========================================"
echo " FocusMamba — Teacher Label Caching"
echo " Array job : ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo " Node      : $(hostname)"
echo " Env       : $ENV"
echo " Cache     : $CACHE_DIR"
echo " Started   : $(date)"
echo "========================================"

nvidia-smi --list-gpus

cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd into $PROJECT_DIR"; exit 1; }
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

source .venv/bin/activate

which uv || { echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
source .venv/bin/activate

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ---------------------------------------------------------------------------
# Cache this environment (both splits; --resume skips already-done clips)
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --config "$CONFIG"
    --cache-dir "$CACHE_DIR"
    --envs "$ENV"
    --num-workers 4
    --batch-size 1
    --resume
)

echo "--- Caching split=train for $ENV ---"
python tools/cache_teacher_labels.py "${COMMON_ARGS[@]}" --split train

echo "--- Caching split=val for $ENV ---"
python tools/cache_teacher_labels.py "${COMMON_ARGS[@]}" --split val

# ---------------------------------------------------------------------------
# Per-task summary
# ---------------------------------------------------------------------------
ENV_CACHE="$CACHE_DIR/$ENV"
N_FILES=$(find "$ENV_CACHE" -name "*.npy" 2>/dev/null | wc -l || echo 0)
DISK=$(du -sh "$ENV_CACHE" 2>/dev/null | cut -f1 || echo "unknown")

echo "========================================"
echo " Done: $ENV"
echo " .npy files : $N_FILES"
echo " Disk used  : $DISK"
echo " Finished   : $(date)"
echo "========================================"
