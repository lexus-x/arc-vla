#!/usr/bin/env bash
# Chain asrc-30K seeds 1 and 2 after seed 0 finishes (only blackwell has the env).
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

wait_done () { for i in $(seq 1 1200); do pgrep -f "train.py .*asrc30k_s0" >/dev/null || break; sleep 60; done; }

run_seed () {
  local SEED=$1 OUT=runs/asrc30k_s${SEED}
  [ -d "$OUT/checkpoints/30000" ] && return 0
  $PY train.py \
    --head deeponet --variant baseline --deeponet_head asrc \
    --dataset lerobot/libero_spatial_image --out "$OUT" \
    --deeponet_p 256 --deeponet_fourier 6 --deeponet_blocks 3 --deeponet_queries 8 \
    --consistency_rates 5,10,25,40,50 --rate_consistency_weight 0.1 \
    --trunk_bandlimit --state_history_steps 8 --warmup 500 \
    --stage1_steps 1650 --stage2_steps 28350 \
    --stage1_batch 48 --stage2_batch 48 \
    --backbone_lr 1e-05 --head_lr 0.0001 --ema 0.999 \
    --num_workers 12 --seed "$SEED" --epoch_steps 200 --ckpt_every 5000 \
    > "train_asrc30k_s${SEED}.log" 2>&1
}

wait_done
run_seed 1
wait_done   # s1 done too
run_seed 2
echo CHAIN_ALL_DONE
