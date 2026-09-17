#!/bin/bash
# campaign 2 (PLAN_BLOCKSUM_HEAD): one (task, seed) stream: 3 heads x {1X,2X,4X}, n=100, suffix _c2
T=$1; S=$2; PY=/home/user/anaconda3/envs/ms3/bin/python; cd "$(dirname "$0")"
run(){ echo "[start] $T $1 s$S k$2 $(date +%H:%M:%S)"; $PY harness.py $T --head $1 --seed $S --k $2 --arms $3 --suffix _c2 > log_dp_${T}_$1_s${S}_k$2_c2.txt 2>&1; echo "[end] rc=$? $(date +%H:%M:%S)"; }
for head in step bspline blocksum; do
  case $head in step) A1=native; A24=zoh,qp;; bspline) A1=native; A24=native;; blocksum) A1=qp,zoh; A24=qp,zoh;; esac
  run $head 1 $A1; run $head 2 $A24  # PushT 4X void by PREREG rule (all arms <10%), skipped
done
