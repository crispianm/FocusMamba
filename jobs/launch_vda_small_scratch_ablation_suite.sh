#!/bin/bash
# =============================================================================
# Launch VDA-small scratch ablation suite (up to 20 concurrent 4-GPU jobs)
# =============================================================================
# Usage:
#   bash jobs/launch_vda_small_scratch_ablation_suite.sh
#
# Optional env vars:
#   RUN_TAG=night1           # default: UTC timestamp
#   MAX_JOBS=20              # submit at most this many experiments
#   PARTITION=workq
#
# This script:
#   1) clones the baseline YAML into run-tagged per-experiment YAMLs
#   2) rewrites log/checkpoint paths per experiment to avoid collisions
#   3) submits each config via sbatch
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BASE_CONFIG="${BASE_CONFIG:-$PROJECT_DIR/configs/experiments/tartanair_vda_small_scratch_distill_full.yaml}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
MAX_JOBS="${MAX_JOBS:-20}"
PARTITION="${PARTITION:-workq}"

GEN_DIR="$PROJECT_DIR/configs/experiments/generated/vda_small_scratch_${RUN_TAG}"
mkdir -p "$GEN_DIR"
mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
source .venv/bin/activate

if [ ! -f "$BASE_CONFIG" ]; then
  echo "ERROR: Baseline config missing: $BASE_CONFIG"
  exit 1
fi

if [ ! -f "$PROJECT_DIR/jobs/run_train_vda_small_scratch_distill.sh" ]; then
  echo "ERROR: Worker SLURM script missing: jobs/run_train_vda_small_scratch_distill.sh"
  exit 1
fi

echo "Generating ablation configs in: $GEN_DIR"
mapfile -t EXP_LINES < <(python - <<'PY' "$BASE_CONFIG" "$GEN_DIR" "$RUN_TAG"
from __future__ import annotations

import copy
import sys
from pathlib import Path
import yaml

base_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
run_tag = sys.argv[3]

with base_path.open("r") as f:
    base = yaml.safe_load(f)

out_dir.mkdir(parents=True, exist_ok=True)

ALL_TEACHERS = ["video_depth_anything", "depth_anything_v3", "depth_pro"]


def keep_teachers(cfg: dict, keep: set[str]) -> None:
    cfg["teachers"] = [t for t in (cfg.get("teachers") or []) if t.get("name") in keep]
    distill = cfg.setdefault("distillation", {})
    distill["teachers"] = [
        t for t in (distill.get("teachers") or []) if t.get("name") in keep
    ]


def set_unique_paths(cfg: dict, exp_name: str) -> None:
    train = cfg.setdefault("training", {})
    train["log_dir"] = f"runs/vda_small_scratch_suite/{run_tag}/{exp_name}"
    train["checkpoint_dir"] = f"checkpoints/vda_small_scratch_suite/{run_tag}/{exp_name}"
    train["verbose_log_file"] = (
        f"runs/vda_small_scratch_suite/{run_tag}/{exp_name}/train_verbose.log"
    )


def make_config(exp_name: str, mutator) -> tuple[str, Path]:
    cfg = copy.deepcopy(base)
    mutator(cfg)
    set_unique_paths(cfg, exp_name)
    out_path = out_dir / f"{exp_name}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return exp_name, out_path


experiments = []

experiments.append(make_config("full_all_teachers", lambda c: None))


def mut_gt_only(c: dict) -> None:
    c["teachers"] = []
    c["distillation"]["enabled"] = False
    c["distillation"]["teachers"] = []
    c["loss"]["distillation_weight"] = 0.0


experiments.append(make_config("gt_only", mut_gt_only))
experiments.append(make_config("no_tgm", lambda c: c["loss"].update({"temporal_weight": 0.0})))
experiments.append(make_config("no_ssi", lambda c: c["loss"].update({"ssi_weight": 0.0})))
experiments.append(
    make_config(
        "no_confidence",
        lambda c: c["distillation"].update({"confidence_weighted": False}),
    )
)
experiments.append(make_config("distill_w025", lambda c: c["loss"].update({"distillation_weight": 0.25})))
experiments.append(make_config("distill_w100", lambda c: c["loss"].update({"distillation_weight": 1.0})))
experiments.append(make_config("distill_w150", lambda c: c["loss"].update({"distillation_weight": 1.5})))


def mut_drop_video(c: dict) -> None:
    keep_teachers(c, set(ALL_TEACHERS) - {"video_depth_anything"})


def mut_drop_da3(c: dict) -> None:
    keep_teachers(c, set(ALL_TEACHERS) - {"depth_anything_v3"})


def mut_drop_depthpro(c: dict) -> None:
    keep_teachers(c, set(ALL_TEACHERS) - {"depth_pro"})


experiments.append(make_config("drop_video_teacher", mut_drop_video))
experiments.append(make_config("drop_da3_teacher", mut_drop_da3))
experiments.append(make_config("drop_depthpro_teacher", mut_drop_depthpro))

experiments.append(make_config("only_video_teacher", lambda c: keep_teachers(c, {"video_depth_anything"})))
experiments.append(make_config("only_da3_teacher", lambda c: keep_teachers(c, {"depth_anything_v3"})))
experiments.append(make_config("only_depthpro_teacher", lambda c: keep_teachers(c, {"depth_pro"})))

experiments.append(make_config("seed_7", lambda c: c["data"].update({"seed": 7})))
experiments.append(make_config("seed_123", lambda c: c["data"].update({"seed": 123})))
experiments.append(make_config("data_hard", lambda c: c["data"].update({"difficulty": "Data_hard"})))
experiments.append(make_config("tgm_w5", lambda c: c["loss"].update({"temporal_weight": 5.0})))
experiments.append(make_config("tgm_w20", lambda c: c["loss"].update({"temporal_weight": 20.0})))
experiments.append(make_config("resolution_336", lambda c: c["data"].update({"image_size": [336, 336]})))

for name, path in experiments:
    print(f"{name}|{path}")
PY
)

TOTAL=${#EXP_LINES[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "ERROR: No configs were generated."
  exit 1
fi

SUBMIT_N="$TOTAL"
if [ "$SUBMIT_N" -gt "$MAX_JOBS" ]; then
  SUBMIT_N="$MAX_JOBS"
fi

echo "Generated $TOTAL experiments; submitting $SUBMIT_N (MAX_JOBS=$MAX_JOBS)."
echo "RUN_TAG=$RUN_TAG"

SUBMITTED=0
for idx in $(seq 0 $((SUBMIT_N - 1))); do
  line="${EXP_LINES[$idx]}"
  EXP_NAME="${line%%|*}"
  CFG_PATH="${line#*|}"

  JOB_NAME="vda_${RUN_TAG}_${EXP_NAME}"
  OUT_LOG="./logs/${JOB_NAME}_%j.out"
  ERR_LOG="./logs/${JOB_NAME}_%j.err"

  echo "Submitting: $EXP_NAME"
  SBATCH_OUT=$(sbatch \
    --partition="$PARTITION" \
    --job-name="$JOB_NAME" \
    --output="$OUT_LOG" \
    --error="$ERR_LOG" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG="$CFG_PATH" \
    jobs/run_train_vda_small_scratch_distill.sh)

  echo "  $SBATCH_OUT"
  SUBMITTED=$((SUBMITTED + 1))
done

echo "========================================"
echo " Submitted jobs : $SUBMITTED"
echo " Config bundle  : $GEN_DIR"
echo " RUN_TAG        : $RUN_TAG"
echo "========================================"
