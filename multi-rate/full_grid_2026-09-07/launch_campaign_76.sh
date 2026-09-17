#!/bin/bash
# Master Campaign Launcher:
# 1. Harvest expanded Push-T dataset (550 demos)
# 2. Launch 537k-step Diffusion training on Push-T and RoboCasa (CloseSingleDoor)
# 3. Launch 30-minute periodic monitor daemon that cascades into Flow Matching across all 3 suites.
set -euo pipefail
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07

MS3=/home/user/anaconda3/envs/ms3/bin/python
GR00T=/home/user/anaconda3/envs/gr00t/bin/python

echo "=== Step 1: Harvesting Large Push-T Dataset (550 demos) ==="
$MS3 harvest_pusht_large.py

echo "=== Step 2: Launching Scaled Diffusion Training (Background) ==="
# Train Push-T with 550 demos
nohup $MS3 -u train_with_checkpoints.py --task PushT-v1 --policy dp --demos demos_PushT-v1_550.npz --steps 537000 > train_dp_pusht_550.log 2>&1 &
echo "[Push-T DP PID: $!]"

# Train RoboCasa CloseSingleDoor with 537k steps
nohup $GR00T -u train_with_checkpoints.py --task RC-CloseSingleDoor --policy dp --demos demos_RC-CloseSingleDoor_39_f0.npz --steps 537000 > train_dp_rc_close.log 2>&1 &
echo "[RoboCasa DP PID: $!]"

echo "=== Step 3: Launching 30-Minute Campaign Monitor & FM Cascade ==="
nohup $MS3 -u campaign_monitor_loop.py --interval 1800 --target_sr 0.76 > campaign_monitor.log 2>&1 &
echo "[Monitor Daemon PID: $!]"

echo "=== CAMPAIGN INITIALIZED SUCCESSFULLY ==="
echo "Logs available at:"
echo "  - Push-T Training: train_dp_pusht_550.log"
echo "  - RoboCasa Training: train_dp_rc_close.log"
echo "  - 30-min Monitor: campaign_progress.log"
