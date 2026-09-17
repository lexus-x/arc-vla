#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=/home/user/anaconda3/envs/gr00t/bin/python
BR=/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python
ARMS="native,tac_fold,tac_fold_satfix,qp,qp_anchor,bspline_eps_raw,bspline"
common=(--policy dp --head step --seed 0 --steps 15000 --n_train 39 --workers 1)

start_bridge() {
  local port=$1
  if ! pgrep -f "robocasa_bridge.py --port $port" >/dev/null; then
    echo "Starting bridge on port $port..."
    "$BR" robocasa_bridge.py --port "$port" >"bridge_${port}_heldout.log" 2>&1 &
    sleep 3
  fi
}

run_eval() {
  local task=$1 k=$2 port=$3
  start_bridge "$port"
  echo "[$(date +%T)] RC-$task heldout n=15 k=$k starting on port $port..."
  "$PY" -u harness.py "RC-$task" "${common[@]}" --n_eval 15 --fold 0 \
    --k "$k" --arms "$ARMS" --port "$port" --suffix _heldout_n15 > "log_rc_${task}_k${k}_heldout.txt" 2>&1
  echo "[$(date +%T)] RC-$task heldout n=15 k=$k FINISHED on port $port"
}

echo "=== Launching parallel k=4 and k=8 for all 4 tasks ==="
run_eval TurnOffSinkFaucet 4 8780 & p1=$!
run_eval TurnOffSinkFaucet 8 8781 & p2=$!
run_eval CoffeePressButton 4 8782 & p3=$!
run_eval CoffeePressButton 8 8783 & p4=$!
run_eval TurnOffMicrowave  4 8784 & p5=$!
run_eval TurnOffMicrowave  8 8785 & p6=$!
run_eval CloseSingleDoor   4 8786 & p7=$!
run_eval CloseSingleDoor   8 8787 & p8=$!

wait "$p1" "$p2" "$p3" "$p4" "$p5" "$p6" "$p7" "$p8"
echo "=== All parallel k=4 and k=8 evaluations completed at $(date) ==="
