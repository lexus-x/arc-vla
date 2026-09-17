#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=/home/user/anaconda3/envs/gr00t/bin/python
BR=/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python
common=(--policy dp --seed 0 --steps 15000 --n_train 39 --n_eval 15 --workers 1)
arms=native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor

# Do not compete with the active full-scale training or the PushT/RoboMimic grid.
while pgrep -f 'robomimic/scripts/train.py --config .*09-15-repro3' >/dev/null || \
      pgrep -f 'run_complete_dp_grid_20260915.sh' >/dev/null; do
  sleep 60
done

start_bridge() {
  local port=$1
  if ! pgrep -f "robocasa_bridge.py --port $port" >/dev/null; then
    "$BR" robocasa_bridge.py --port "$port" >"bridge_${port}_completegrid.log" 2>&1 &
    sleep 10
  fi
}

for port in 8765 8766 8767 8768; do start_bridge "$port"; done

run_paper_task() {
  local task=$1 port=$2 fold k
  for fold in 0 1 2; do
    for k in 1 2 4 8; do
      "$PY" -u harness.py "RC-$task" "${common[@]}" --head step --fold "$fold" --port "$port" \
        --k "$k" --arms "$arms" --suffix _completegrid
    done
    for k in 2 4 8; do
      "$PY" -u harness.py "RC-$task" "${common[@]}" --head bspline --fold "$fold" --port "$port" \
        --k "$k" --arms native --suffix _completegrid
    done
  done
}

run_paper_task TurnOffSinkFaucet 8765 &
p1=$!
run_paper_task CoffeePressButton 8766 &
p2=$!
run_paper_task TurnOffMicrowave 8767 &
p3=$!
run_paper_task CloseSingleDoor 8768 &
p4=$!
wait "$p1" "$p2" "$p3" "$p4"

# Legacy state tasks have no trained B-spline-head checkpoints; evaluate step-head methods only.
run_legacy_task() {
  local task=$1 port=$2 k
  for k in 1 2 4 8; do
    "$PY" -u harness.py "RC-$task" "${common[@]}" --head step --port "$port" \
      --k "$k" --arms "$arms" --suffix _completegrid
  done
}
run_legacy_task OpenDrawer 8765 &
p1=$!
run_legacy_task PnPCounterToStove 8766 &
p2=$!
wait "$p1" "$p2"
