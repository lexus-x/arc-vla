#!/bin/bash
set -e
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
export ARMS="original,exact_integral,cubic_spline_satfix,tac_fold_satfix,bspline_eps_satfix,spline,bspline_eps_raw,qp,qp_anchor"
export OUT_PREFIX="qpanchor_fair_replay"
# Pinned to 12 cores + thread-pool caps: the RC eval grid (run_robocasa_eval_grid_20260916.sh,
# n=100 phase, hours-long, already in flight) was starved to 57 load-avg/56 cores by this job's
# unbounded physx_cpu thread spawn (173 threads at default affinity) on first launch. taskset
# gives every spawned thread the restricted mask from birth, unlike a post-hoc renice.
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
for K in 2 3 4; do
  echo "[campaign] starting k=$K at $(date)"
  taskset -c 0-11 conda run -n ms3 python3 run_qp_replay.py PickCube-v1 100000 $K > "log_qpanchor_fair_k${K}.txt" 2>&1
  echo "[campaign] finished k=$K at $(date)"
done
echo "[campaign] ALL DONE $(date)"
