#!/bin/bash
set -e
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
export ARMS="original,exact_integral,cubic_spline_satfix,tac_fold_satfix,bspline_eps_satfix,spline,bspline_eps_raw,qp,qp_anchor"
export OUT_PREFIX="pusht_fair_replay"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
for K in 2 3 4; do
  echo "[campaign] starting k=$K at $(date)"
  taskset -c 0-11 conda run -n ms3 python3 run_qp_replay.py PushT-v1 100000 $K > "log_pusht_fair_k${K}.txt" 2>&1
  echo "[campaign] finished k=$K at $(date)"
done
echo "[campaign] ALL DONE $(date)"
