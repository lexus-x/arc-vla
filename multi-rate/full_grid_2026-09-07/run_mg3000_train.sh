#!/usr/bin/env bash
set -e

PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
ROBOMIMIC=/tmp/claude-1000/-home-user-Desktop/eca0093a-6f27-4f08-b55b-27d2b8c90be8/scratchpad/robomimic
ROBOCASA=/home/user/Isaac-GR00T/external_dependencies/robocasa
export PYTHONPATH="$ROBOMIMIC:$ROBOCASA"

CFG_BASE="$ROBOMIMIC/expdata/robocasa/diffusion_policy/09-15-repro3"

declare -A TASKS=(
  ["TurnOffSinkFaucet"]="$CFG_BASE/seed_123_ds_TurnOffSinkFaucet/20260915233625/config.json"
  ["CoffeePressButton"]="$CFG_BASE/seed_123_ds_CoffeePressButton/20260915233631/config.json"
  ["TurnOffMicrowave"]="$CFG_BASE/seed_123_ds_TurnOffMicrowave/20260915233637/config.json"
  ["CloseSingleDoor"]="$CFG_BASE/seed_123_ds_CloseSingleDoor/20260915233643/config.json"
)

LOG_DIR="/home/user/Desktop/multi-rate/full_grid_2026-09-07"

for task in TurnOffSinkFaucet CoffeePressButton TurnOffMicrowave CloseSingleDoor; do
  cfg="${TASKS[$task]}"
  log="$LOG_DIR/train_mg3000_${task}.log"
  echo "Launching $task with config $cfg -> $log"
  nohup "$PY" "$ROBOMIMIC/robomimic/scripts/train.py" --config "$cfg" > "$log" 2>&1 &
done

echo "All 4 RoboCasa 3,000-demo training runs launched successfully."
