#!/usr/bin/env bash
# Stage-1 of the clean<->degraded consistency prototype (local RTX 4090).
# Sequential queue: for each arm, train (6 epochs, WDDA, 5e-5, saves epoch_*.pt
# ladder) then run the severity x seed sweep on its ladder. Designed to survive
# the work machine sleeping (run inside tmux):
#
#   tmux new -s cc 'bash clean_consistency_prototype/scripts/run_stage1.sh'
#
set -u
cd /mnt/DATA/DegradedDepth/FocusMamba
PY=~/.venvs/focusmamba/bin/python
P=clean_consistency_prototype
SWEEP=degraded_erosion_investigation/scripts/eval_severity_sweep.py
LOG=$P/runs/stage1_driver.log
mkdir -p "$P/runs"
ARMS=(c0_control si_w0p5 si_w1p0 si_w2p0 logl1_w0p5 logl1_w1p0 logl1_w2p0)

echo "=== Stage-1 queue start $(date) ===" | tee -a "$LOG"
for NAME in "${ARMS[@]}"; do
    CFG=$P/configs/$NAME.yaml
    CKPT=$P/ckpts/$NAME
    echo "=== START $NAME $(date) ===" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 $PY train.py --config "$CFG" --verbose
    rc=$?
    echo "=== END $NAME rc=$rc $(date) ===" | tee -a "$LOG"
    if [ "$rc" -ne 0 ]; then
        echo "WARN $NAME training rc=$rc — continuing to next arm" | tee -a "$LOG"
        continue
    fi
    # severity x seed sweep on the per-epoch ladder (VKITTI primary; TartanAir too)
    for DS in vkitti tartanair; do
        CUDA_VISIBLE_DEVICES=0 $PY "$SWEEP" \
            --config "$CFG" \
            --checkpoint-dir "$CKPT" \
            --baseline-checkpoint checkpoints/metric_video_depth_anything_vits.pth \
            --dataset "$DS" \
            --severities 0.0,0.25,0.5,0.75,1.0 \
            --seeds 1,2,3 \
            --max-clips 40 --batch-size 2 \
            --out "$P/runs/$NAME/sweep_${DS}.csv" \
            || echo "WARN sweep $NAME/$DS failed" | tee -a "$LOG"
    done
done
echo "=== Stage-1 queue done $(date) ===" | tee -a "$LOG"
