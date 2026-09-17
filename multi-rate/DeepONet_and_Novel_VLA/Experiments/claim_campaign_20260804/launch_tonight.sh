#!/usr/bin/env bash
# 2026-08-21 rate-paper launch: asrc-30K training + 10Hz k-sweep + native 520/5 ceiling.
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

# 1) asrc 30K-step training, seed 0 (matches flow30 budget)
nohup $PY train.py \
  --head deeponet --variant baseline --deeponet_head asrc \
  --dataset lerobot/libero_spatial_image --out runs/asrc30k_s0 \
  --deeponet_p 256 --deeponet_fourier 6 --deeponet_blocks 3 --deeponet_queries 8 \
  --consistency_rates 5,10,25,40,50 --rate_consistency_weight 0.1 \
  --trunk_bandlimit --state_history_steps 8 --warmup 500 \
  --stage1_steps 1650 --stage2_steps 28350 \
  --stage1_batch 48 --stage2_batch 48 \
  --backbone_lr 1e-05 --head_lr 0.0001 --ema 0.999 \
  --num_workers 12 --seed 0 --epoch_steps 200 --ckpt_every 5000 \
  > train_asrc30k_s0.log 2>&1 &
echo "training pid $!"

# 2) 10 Hz magscale k-sweep, 4 procs in parallel, n=50 each (k=1.0 is the no-rescale control)
for K in 0.3 0.5 0.7 1.0; do
  TAG="$(echo $K | tr '.' 'p')"
  MAGSCALE_K=$K $PY evaluate_multirate_honest.py \
    --arms asrc_mag_folding_10env --trials_per_task 5 --suite libero_spatial \
    --out "ksweep_k${TAG}_out" > "ksweep_k${TAG}.log" 2>&1 &
done
echo "ksweep launched"

# 3) asrc native ceiling at the 520-step / replan-5 protocol (fair vs flow30 81.7)
nohup $PY evaluate_multirate_honest.py \
  --arms asrc_native_20env --trials_per_task 30 --suite libero_spatial \
  --wallclock 26.0 --replan_steps 5 \
  --out r5_asrc_native_20env_out > r5_asrc_native_20env.log 2>&1 &
echo "native520 pid $!"

sleep 5
echo "--- procs ---"
pgrep -af "train.py|evaluate_multirate_honest.py" | head -10
echo LAUNCHED
