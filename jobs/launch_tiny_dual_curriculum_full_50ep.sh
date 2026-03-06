#!/bin/bash
# Submit strict 2-stage tiny dual-head curriculum:
#   Stage A: relative depth pretraining
#   Stage B: metric fine-tuning resumed from Stage A best checkpoint

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/b5dh/FocusMamba}"
RUNNER="${RUNNER:-jobs/run_train_vda_small_scratch_distill.sh}"
STAGE_A_CONFIG="${STAGE_A_CONFIG:-configs/experiments/tartanair_vda_tiny_dual_dinov3_stageA_relative.yaml}"
STAGE_B_CONFIG="${STAGE_B_CONFIG:-configs/experiments/tartanair_vda_tiny_dual_dinov3_stageB_metric.yaml}"

cd "$PROJECT_DIR"
mkdir -p logs

for cfg in "$STAGE_A_CONFIG" "$STAGE_B_CONFIG"; do
  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: missing config: $cfg"
    exit 1
  fi
done

readarray -t meta < <(source .venv/bin/activate && python - <<'PY' "$STAGE_A_CONFIG" "$STAGE_B_CONFIG"
import sys
import yaml

def get_ckpt(path: str) -> str:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return str((cfg.get("training", {}) or {}).get("checkpoint_dir", ""))

print(get_ckpt(sys.argv[1]))
print(get_ckpt(sys.argv[2]))
PY
)

ckpt_a="${meta[0]}"
ckpt_b="${meta[1]}"
if [[ -z "$ckpt_a" || -z "$ckpt_b" ]]; then
  echo "ERROR: checkpoint_dir missing in one of the configs."
  exit 1
fi

resume_from="${PROJECT_DIR}/${ckpt_a}/best.pt"

j_a=$(sbatch --parsable \
  --job-name=tnyA_rel20 \
  --output="./logs/tnyA_rel20_%j.out" \
  --error="./logs/tnyA_rel20_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG="$STAGE_A_CONFIG",RESET_CHECKPOINTS=1 \
  "$RUNNER")

j_b=$(sbatch --parsable \
  --dependency="afterok:${j_a}" \
  --job-name=tnyB_met30 \
  --output="./logs/tnyB_met30_%j.out" \
  --error="./logs/tnyB_met30_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG="$STAGE_B_CONFIG",RESET_CHECKPOINTS=1,RESUME_FROM="$resume_from" \
  "$RUNNER")

echo "Submitted strict 2-stage curriculum:"
echo "  Stage A job: $j_a  ($STAGE_A_CONFIG)"
echo "  Stage B job: $j_b  ($STAGE_B_CONFIG) afterok:$j_a"
echo "  Stage B resume source: $resume_from"
echo
echo "Queue snapshot:"
squeue -u "$USER" -o '%.18i %.9P %.18j %.8T %.10M %.6D %R'
