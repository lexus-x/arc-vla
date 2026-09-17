#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=/home/user/anaconda3/envs/gr00t/bin/python
arms=native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor

run_task() {
  local task=$1 port=$2 k
  for k in 1 2 4 8; do
    "$PY" -u harness.py "RC-$task" --policy dp --head step --fold 0 \
      --n_train 39 --n_eval 100 --steps 15000 --k "$k" --arms "$arms" \
      --suffix _random_n100 --port "$port" --workers 1 --rc-random-eval --eval-seed 0
  done
}

run_task TurnOffSinkFaucet 8770 & p1=$!
run_task CoffeePressButton 8771 & p2=$!
run_task TurnOffMicrowave 8772 & p3=$!
run_task CloseSingleDoor 8773 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"
