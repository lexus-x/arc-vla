#!/bin/bash
# DP seed replicates. Protocol identical to full_grid_2026-09-07 (same code, copied frozen);
# only --seed changes. ManiSkill only -- PushT/PickCube are the sole tasks with a real effect
# (FIRST_PRINCIPLES.md S2/S3); RoboMimic/RoboCasa are at ceiling/floor, extra seeds buy nothing.
set -u
cd /home/user/Desktop/multi-rate/seeds_2026-09-07
MS3=/home/user/anaconda3/envs/ms3/bin/python

run() {
  task=$1; seed=$2
  tag="dp_${task}_s${seed}_30k"
  [ -f "result_${tag}.json" ] && { echo "[skip] $tag done $(date +%H:%M:%S)"; return; }
  echo "[start] $tag $(date +%H:%M:%S)"
  $MS3 harness.py "$task" --policy dp --seed "$seed" > "log_${tag}.txt" 2>&1
  echo "[end] $tag rc=$? $(date +%H:%M:%S)"
}

( run PushT-v1    1; run PushT-v1    2 ) &
( run PickCube-v1 1; run PickCube-v1 2 ) &
wait
echo "[campaign] ALL DONE $(date +%H:%M:%S)"
