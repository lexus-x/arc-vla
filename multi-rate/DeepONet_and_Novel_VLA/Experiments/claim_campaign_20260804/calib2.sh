#!/bin/bash
# Re-run the calibration with state history corrected (v2 -> 1, not 8).
# Original harness: 85.0% at replan=5 / 520 steps. Previous attempt here: 31.0%.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=6
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
$PY evaluate_multirate_honest.py --arms v2_spatial_native_20env --trials_per_task 20     --suite libero_spatial --replan_steps 5 --wallclock 26.0 --out calib2_out     > calib2.log 2>&1
echo "[calib2] $(date +%H:%M) v2 statehist=1 replan=5 exit=$?" >> final8.log
