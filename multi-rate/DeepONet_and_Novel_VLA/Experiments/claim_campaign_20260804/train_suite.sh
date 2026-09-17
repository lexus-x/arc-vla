#!/bin/bash
# Train asrc on one LIBERO suite, replicating asrc_s0's config EXACTLY except --dataset/--out.
# Usage: ./train_suite.sh <dataset_id> <out_dir>
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
DS="$1"; OUT="$2"
export CUDA_VISIBLE_DEVICES=0
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

exec $PY train.py \
  --head deeponet --variant baseline --deeponet_head asrc \
  --dataset "$DS" --out "$OUT" \
  --deeponet_p 256 --deeponet_fourier 6 --deeponet_blocks 3 --deeponet_queries 8 \
  --consistency_rates 5,10,25,40,50 --rate_consistency_weight 0.1 \
  --trunk_bandlimit \
  --state_history_steps 8 --warmup 500 \
  --stage1_steps 1650 --stage2_steps 6650 \
  --stage1_batch 48 --stage2_batch 48 \
  --backbone_lr 1e-05 --head_lr 0.0001 --ema 0.999 \
  --num_workers 12 --seed 0 --epoch_steps 200 --ckpt_every 2000
