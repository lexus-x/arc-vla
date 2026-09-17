#!/bin/bash
# B-spline Policy Table 2a RoboCasa tasks: one task per stream/bridge port. 3 folds x {step head (3 arms), bspline head}.
T=$1; P=$2; PY=/home/user/anaconda3/envs/gr00t/bin/python; cd "$(dirname "$0")"
run(){ echo "[start] $T $1 f$2 $(date +%H:%M:%S)"; $PY harness.py RC-$T --head $1 --fold $2 --n_train 39 --n_eval 15 --steps 15000 --k 1 --arms $3 --suffix _rc4 --port $P > log_dp_RC-${T}_$1_f$2_rc4.txt 2>&1; echo "[end] rc=$? $(date +%H:%M:%S)"; }
for f in 0 1 2; do run step $f native,qp_eps05,bspline_eps05_raw; run bspline $f native; done
