#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=/home/user/anaconda3/envs/gr00t/bin/python
BR=/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python
# 7-arm list fixed by PREREG_ROBOCASA_RANDOM_N100.md / PREREG_COMPLETE_DP_GRID_2026-09-15.md:31-32.
# Dropped from the 2026-09-16 run: unregistered raw `tac_fold`, and `bspline` (scipy splprep/splev
# mismatch -- splev evaluated at spline parameter u, not at time t; deviates 1.35 from identity at
# k=1). bspline_eps_raw is the correct Alg.1 implementation and stays.
ARMS="native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor"
common=(--policy dp --head step --seed 0 --steps 15000 --n_train 39 --workers 1)

start_bridge() {
  local port=$1
  if ! pgrep -f "robocasa_bridge.py --port $port" >/dev/null; then
    echo "Starting bridge on port $port..."
    "$BR" robocasa_bridge.py --port "$port" >"bridge_${port}_usergrid.log" 2>&1 &
    sleep 5
  fi
}

for port in 8765 8766 8767 8768; do
  start_bridge "$port"
done

# Task runners: held-out demo states (n=15)
run_task_heldout() {
  local task=$1 port=$2
  # k=1 skipped: resamplers are the identity map at k=1 (max|arm-native| ~3e-08 for zoh/spline;
  # existing result_dp_RC-*_f0_k1_heldout_n15.json already confirm native==qp==qp_anchor==
  # bspline_eps_raw exactly on all four tasks). Re-running would also overwrite the only files
  # holding the tac_fold/bspline k=1 columns -- see k1_heldout_backup_20260916/.
  for k in 2 4 8; do
    echo "[$(date +%T)] $task heldout n=15 k=$k starting..."
    "$PY" -u harness.py "RC-$task" "${common[@]}" --n_eval 15 --fold 0 \
      --k "$k" --arms "$ARMS" --port "$port" --suffix _heldout_n15
  done
}

# Task runners: random resets (n=100). k=1 skipped: seeds 1e6..1e6+99 and arms are byte-identical
# to the existing result_dp_RC-*_f0_k1_random_n100.json (already the correct 7-arm set).
run_task_random_n100() {
  local task=$1 port=$2
  for k in 2 4 8; do
    echo "[$(date +%T)] $task random n=100 k=$k starting..."
    "$PY" -u harness.py "RC-$task" "${common[@]}" --n_eval 100 --fold 0 \
      --k "$k" --arms "$ARMS" --port "$port" --rc-random-eval --eval-seed 0 --suffix _random_n100
  done
}

# Step 1: Run heldout n=15 in parallel across 4 bridges
echo "=== Phase 1: Held-out demo evaluation (n=15) across 4 tasks ==="
run_task_heldout TurnOffSinkFaucet 8765 > log_rc_faucet_heldout.txt 2>&1 & p1=$!
run_task_heldout CoffeePressButton 8766 > log_rc_coffee_heldout.txt 2>&1 & p2=$!
run_task_heldout TurnOffMicrowave  8767 > log_rc_micro_heldout.txt  2>&1 & p3=$!
run_task_heldout CloseSingleDoor   8768 > log_rc_close_heldout.txt  2>&1 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"
echo "=== Phase 1 finished at $(date) ==="

# Step 2: Run random n=100 in parallel across 4 bridges
echo "=== Phase 2: Random reset evaluation (n=100) across 4 tasks ==="
run_task_random_n100 TurnOffSinkFaucet 8765 > log_rc_faucet_random.txt 2>&1 & p1=$!
run_task_random_n100 CoffeePressButton 8766 > log_rc_coffee_random.txt 2>&1 & p2=$!
run_task_random_n100 TurnOffMicrowave  8767 > log_rc_micro_random.txt  2>&1 & p3=$!
run_task_random_n100 CloseSingleDoor   8768 > log_rc_close_random.txt  2>&1 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"
echo "=== Phase 2 finished at $(date) ==="
