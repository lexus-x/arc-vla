#!/bin/bash
set -e
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
export ARMS="original,exact_integral,cubic_spline_satfix,tac_fold_satfix,bspline_eps_satfix,qp,qp_anchor"
export OUT_PREFIX="qpanchor_replay"
for K in 2 3 4; do
  echo "[campaign] starting k=$K at $(date)"
  conda run -n ms3 python3 run_qp_replay.py PickCube-v1 100000 $K > "log_qpanchor_k${K}.txt" 2>&1
  echo "[campaign] finished k=$K at $(date)"
done
echo "[campaign] ALL DONE $(date)"
