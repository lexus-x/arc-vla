#!/usr/bin/env bash
# PREREG_ARC_4BENCH.md Phase 1: pair spline/bspline against sealed confirm windows (k=4)
set -u
PY=/home/user/anaconda3/envs/ms3/bin/python
for T in RollBall-v1 LiftPegUpright-v1 PushCube-v1 AnymalC-Reach-v1 PokeCube-v1 StackCube-v1; do
  echo "[$(date +%T)] START $T"
  $PY harness.py "$T" --k 4 --n_train 200 --n_eval 400 --eval-offset 100 --seed 0 --workers 4 \
      --arms native,zoh,spline,spline_satfix,bspline_eps_raw,qp_anchor --suffix _splinefaceoff \
      > "faceoff_${T}_k4.log" 2>&1
  echo "[$(date +%T)] DONE  $T rc=$?"
done
echo ALL_DONE
