#!/bin/bash
set -e
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
PY=/home/user/anaconda3/envs/ms3/bin/python
ARMS="native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor"
for K in 2 3; do
  echo "[campaign] starting k=$K at $(date)"
  "$PY" -u harness.py PushT-v1 --policy dp --steps 30000 --seed 0 --k "$K" --n_eval 400 --n_train 200 \
    --arms "$ARMS" --suffix _fair_n400 > "log_pusht_cl_fair_k${K}.txt" 2>&1
  echo "[campaign] finished k=$K at $(date)"
done
echo "[campaign] ALL DONE $(date)"
