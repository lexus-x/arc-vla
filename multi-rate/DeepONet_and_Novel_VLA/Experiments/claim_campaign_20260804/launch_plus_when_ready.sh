#!/usr/bin/env bash
# Auto-launch the matched-8300 Plus eval when flow@8300 training finishes.
CK=/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300/model.safetensors
CAMP=/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804
while [ ! -f "$CK" ]; do sleep 60; done
echo "checkpoint ready $(date)"
sleep 30
cd "$CAMP" || exit 1
nohup bash run_matched8300_plus.sh > plus_matched8300_run.log 2>&1 &
echo "PLUS_EVAL_LAUNCHED pid $! at $(date)"
