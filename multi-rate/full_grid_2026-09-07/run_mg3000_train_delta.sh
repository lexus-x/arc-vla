#!/usr/bin/env bash
set -e
# Relaunch of run_mg3000_train.sh with 3 fixes applied to the configs (not this script):
#   action_keys: actions_abs -> actions (delta, resampler/harness-compatible)
#   experiment.rollout.enabled: true (robomimic's own native rollout eval, piggybacks on the
#     existing every-100-epoch cadence -- sidesteps extending robocasa_bridge.py/harness.py
#     for image obs, which is real, un-scoped work; see TABLE_COMPLETE_DP_GRID_LIVE.md:32)
#   train.output_dir -> /media/user/C2FE578FFE577A9D (384G free, not the 39G-free main disk,
#     not another session's /tmp scratchpad)
# Also: save.every_n_epochs 100 -> 10, so a checkpoint exists long before another idle-gap kill.

PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
ROBOMIMIC=/tmp/claude-1000/-home-user-Desktop/eca0093a-6f27-4f08-b55b-27d2b8c90be8/scratchpad/robomimic
ROBOCASA=/home/user/Isaac-GR00T/external_dependencies/robocasa
export PYTHONPATH="$ROBOMIMIC:$ROBOCASA"

CFG_DIR="/home/user/Desktop/multi-rate/full_grid_2026-09-07/robocasa_mg3000_configs"
LOG_DIR="/home/user/Desktop/multi-rate/full_grid_2026-09-07"

pids=()
for task in TurnOffSinkFaucet CoffeePressButton TurnOffMicrowave CloseSingleDoor; do
  cfg="$CFG_DIR/${task}.json"
  log="$LOG_DIR/train_mg3000_${task}_delta_restart_20260917.log"
  echo "Launching $task with config $cfg -> $log"
  "$PY" "$ROBOMIMIC/robomimic/scripts/train.py" --config "$cfg" > "$log" 2>&1 &
  pids+=("$!")
done

echo "All 4 RoboCasa 3,000-demo delta-action training runs launched successfully."
trap 'kill -TERM "${pids[@]}" 2>/dev/null || true' TERM INT
wait "${pids[@]}"
