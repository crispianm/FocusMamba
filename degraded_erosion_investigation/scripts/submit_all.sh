#!/bin/bash
# Submit the degraded-erosion investigation jobs on Isambard (run from repo root).
# base_small first (Phase B), then the C1-C5 ablations (Phase C). Each is one
# 1-GPU short job that trains + runs the severity/seed sweep on its ladder.
set -euo pipefail
cd /projects/b5dh/FocusMamba
mkdir -p logs
for NAME in base_small c1_no_ema c2_no_tgm c3_low_lr c4_clean_mix c5_severe_curriculum; do
    jid=$(sbatch --parsable jobs/erosion_run.sh "$NAME")
    echo "submitted $NAME -> job $jid"
done
squeue --me
