#!/bin/bash
# CALIBRATION ONLY -- not a protocol choice.
# Reproduce the original authors' exact Spatial setting for v2: replan=5, max_steps=520.
# They reported 85.0%. If this harness returns ~85% the instrument is sound and replan is a real
# cliff. If it returns ~30%, every v2 number from this harness today is void.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
$PY evaluate_multirate_honest.py --arms v2_spatial_native_20env --trials_per_task 20     --suite libero_spatial --replan_steps 5 --wallclock 26.0 --out calib_v2_r5_out     > calib_v2_r5.log 2>&1
echo "[calib] $(date +%H:%M) v2 replan=5 wc=26s(520 steps) exit=$?" >> final8.log
