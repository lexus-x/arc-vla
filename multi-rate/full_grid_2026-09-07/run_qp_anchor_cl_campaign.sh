#!/bin/bash
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
ARMS="native,zoh,spline_satfix,tac_fold_satfix,bspline_eps_satfix,qp,qp_anchor"
echo "[cl_campaign] starting DP PickCube k=4 n=400 at $(date)"
conda run -n ms3 python3 harness.py PickCube-v1 --policy dp --k 4 --n_eval 400 --arms "$ARMS" --suffix _n400_anchor > log_dp_PickCube-v1_k4_n400_anchor.txt 2>&1
echo "[cl_campaign] finished DP at $(date)"
echo "[cl_campaign] starting FM PickCube k=4 n=100 at $(date)"
conda run -n ms3 python3 harness.py PickCube-v1 --policy fm --k 4 --n_eval 100 --arms "$ARMS" --suffix _anchor > log_fm_PickCube-v1_k4_anchor.txt 2>&1
echo "[cl_campaign] finished FM at $(date)"
echo "[cl_campaign] ALL DONE $(date)"
