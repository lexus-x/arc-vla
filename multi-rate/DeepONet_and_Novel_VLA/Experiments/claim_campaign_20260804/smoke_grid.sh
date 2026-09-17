#!/bin/bash
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=6
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
rm -rf smoke_v2_out smoke_fl_out
$PY evaluate_multirate_honest.py --arms v2_spatial_native_20env --trials_per_task 1    --suite libero_spatial --out smoke_v2_out > smoke_v2.log 2>&1
echo "v2 exit=$?" > smoke_grid.done
$PY evaluate_multirate_honest.py --arms flow30_long_native_20env --trials_per_task 1    --suite libero_10 --out smoke_fl_out > smoke_fl.log 2>&1
echo "flow30 exit=$?" >> smoke_grid.done
