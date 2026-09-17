#!/bin/bash
set -e
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
export ARMS="original,exact_integral,cubic_spline_satfix,tac_fold_satfix,bspline_eps_satfix,spline,bspline_eps_raw,qp,qp_anchor"
export OUT_PREFIX="pusht_fair_replay"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
# resume: k=2 already completed (void result, original itself only 0.28% -- see report).
# k=3/k=4 died silently ~23:12-23:26 when the other session's RoboCasa training landed on GPU;
# this job is CPU-only (physx_cpu) and never touched GPU, so it's safe to relaunch now even
# though that training is still running -- CPU headroom is genuinely free (load ~21/56).
for K in 3 4; do
  echo "[resume] starting k=$K at $(date)"
  taskset -c 0-11 conda run -n ms3 python3 run_qp_replay.py PushT-v1 100000 $K > "log_pusht_fair_k${K}.txt" 2>&1
  echo "[resume] finished k=$K at $(date)"
done
echo "[resume] ALL DONE $(date)"
