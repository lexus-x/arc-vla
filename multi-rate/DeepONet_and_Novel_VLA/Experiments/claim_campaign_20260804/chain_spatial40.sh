#!/bin/bash
# Wait for the Plus@40 run to finish, then launch Spatial@40 paired n=300.
# NEVER concurrent: MuJoCo + dataloaders saturate CPU and both runs crawl.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804
while pgrep -f evaluate_plus_multirate >/dev/null; do sleep 60; done
echo "[chain] Plus@40 finished $(date). Starting Spatial@40." >> chain.log
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl LIBERO_SUITE=libero_spatial
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
$PY evaluate_multirate_honest.py \
  --arms asrc_cadmag_folding_40env,asrc_anchor_40env,asrc_cadmag_spline_40env \
  --trials_per_task 30 --suite libero_spatial --out spatial40_out \
  > spatial40.log 2>&1
echo "[chain] Spatial@40 finished $(date)." >> chain.log
