#!/bin/bash
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
A=v2_spatial_cadmag_spline_40env
$PY evaluate_multirate_honest.py --arms $A --trials_per_task 30 --suite libero_spatial     --out grid_${A}_out > grid_${A}.log 2>&1
echo "[q6] $(date +%H:%M) REDO $A exit=$?" >> final6.log
