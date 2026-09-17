#!/usr/bin/env bash
# Round 2 (2026-08-21 ~15:20): attack the 10 Hz gap.
# A) 10Hz-weighted consistency fine-tune from asrc_s0 (head-only, 2000 steps)
# B) replan sensitivity @10Hz k=0.5: replan 2 (0.2s) and replan 10 (1.0s), n=50 each
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

nohup $PY train.py \
  --head deeponet --variant baseline --deeponet_head asrc \
  --base_ckpt runs/asrc_s0/checkpoints/8300 \
  --dataset lerobot/libero_spatial_image --out runs/asrc10hz_ft_s0 \
  --deeponet_p 256 --deeponet_fourier 6 --deeponet_blocks 3 --deeponet_queries 8 \
  --consistency_rates 10 --rate_consistency_weight 0.2 \
  --trunk_bandlimit --state_history_steps 8 --warmup 200 \
  --stage1_steps 2000 --stage2_steps 0 --stage1_batch 48 \
  --head_lr 1e-4 --ema 0.999 --num_workers 8 --seed 0 \
  --epoch_steps 200 --ckpt_every 2000 \
  > train_asrc10hz_ft_s0.log 2>&1 &
echo "ft pid $!"

MAGSCALE_K=0.5 $PY evaluate_multirate_honest.py \
  --arms asrc_mag_folding_10env --trials_per_task 5 --suite libero_spatial \
  --replan_steps 2 --out rp2_10hz_k05_out > rp2_10hz_k05.log 2>&1 &
echo "rp2 pid $!"

MAGSCALE_K=0.5 $PY evaluate_multirate_honest.py \
  --arms asrc_mag_folding_10env --trials_per_task 5 --suite libero_spatial \
  --replan_steps 10 --out rp10_10hz_k05_out > rp10_10hz_k05.log 2>&1 &
echo "rp10 pid $!"

sleep 3
pgrep -af "train.py|evaluate_multirate_honest.py" | wc -l
echo ROUND2_LAUNCHED
