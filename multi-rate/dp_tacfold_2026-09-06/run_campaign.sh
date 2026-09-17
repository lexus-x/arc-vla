#!/bin/bash
# Two parallel streams, sequential within stream; skip-if-done via result_<task>.json; checkpoints reused via dp_<task>.pt
cd /home/user/Desktop/multi-rate/dp_tacfold_2026-09-06
run() { py=$1; task=$2; [ -f "result_${task}.json" ] && { echo "[skip] $task done"; return; }
        echo "[start] $task $(date +%H:%M:%S)"; $py harness.py "$task" > "log_${task}.txt" 2>&1; echo "[end] $task rc=$? $(date +%H:%M:%S)"; }
( run /home/user/anaconda3/envs/ms3/bin/python PushT-v1; run /home/user/anaconda3/envs/ms3/bin/python PickCube-v1 ) &
( run /home/user/anaconda3/envs/vla_smolvla_libero/bin/python lift; run /home/user/anaconda3/envs/vla_smolvla_libero/bin/python can; run /home/user/anaconda3/envs/vla_smolvla_libero/bin/python square ) &
wait; echo "[campaign] ALL DONE $(date +%H:%M:%S)"
