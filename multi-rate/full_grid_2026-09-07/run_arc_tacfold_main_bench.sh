#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
MS_PY=${ARC_MS_PY:-/home/user/anaconda3/envs/ms3/bin/python}
RM_PY=${ARC_RM_PY:-/home/user/anaconda3/envs/vla_smolvla_libero/bin/python}
RC_PY=${ARC_RC_PY:-/home/user/anaconda3/envs/gr00t/bin/python}
BR=${ARC_BRIDGE_PY:-/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python}
VISION_DATA=${ARC_VISION_DATA:-"$PWD/robocasa_data_vision"}
VISION_OUT=${ARC_VISION_OUT:-"$PWD/stage1_arc_vision_results"}
RATES=(1 2 4)

export NUMBA_CACHE_DIR=${ARC_NUMBA_CACHE_DIR:-"$PWD/.arc_numba_cache"}
mkdir -p "$NUMBA_CACHE_DIR"

for executable in "$MS_PY" "$RM_PY" "$RC_PY" "$BR"; do
  [[ -x $executable ]] || { echo "Missing executable: $executable" >&2; exit 2; }
done

run_pair() {
  local py=$1 task=$2 n_train=$3 n_eval=$4 steps=$5 k=$6
  "$py" -u harness.py "$task" --head step --n_train "$n_train" --n_eval "$n_eval" \
    --steps "$steps" --k "$k" --arms native,zoh,spline,bspline_eps_raw,tac_fold_satfix \
    --suffix _arcmain_ref \
    --checkpoint-suffix _arcmain_ref
  "$py" -u harness.py "$task" --head arc --n_train "$n_train" --n_eval "$n_eval" \
    --steps "$steps" --k "$k" --arms zoh,spline,bspline_eps_raw,tac_fold,tac_fold_satfix \
    --suffix _arcmain \
    --checkpoint-suffix _arcmain
}

for k in "${RATES[@]}"; do run_pair "$MS_PY" PushT-v1 200 400 30000 "$k"; done
for task in lift can square; do
  for k in "${RATES[@]}"; do run_pair "$RM_PY" "$task" 200 100 30000 "$k"; done
done

start_bridge() {
  local port=$1
  pgrep -f "robocasa_bridge.py --port $port" >/dev/null || {
    "$BR" robocasa_bridge.py --port "$port" >"bridge_${port}_arcmain.log" 2>&1 &
    sleep 5
  }
}
for port in 8765 8766 8767 8768; do start_bridge "$port"; done

run_rc() {
  local task=$1 port=$2 k
  for k in "${RATES[@]}"; do
    "$RC_PY" -u harness.py "RC-$task" --policy dp --head step --fold 0 --port "$port" \
      --n_train 39 --n_eval 100 --steps 15000 --k "$k" \
      --arms native,zoh,spline,bspline_eps_raw,tac_fold_satfix \
      --rc-random-eval --eval-seed 0 --suffix _arcmain_ref --checkpoint-suffix _arcmain_ref
    "$RC_PY" -u harness.py "RC-$task" --policy dp --head arc --fold 0 --port "$port" \
      --n_train 39 --n_eval 100 --steps 15000 --k "$k" \
      --arms zoh,spline,bspline_eps_raw,tac_fold,tac_fold_satfix \
      --rc-random-eval --eval-seed 0 --suffix _arcmain --checkpoint-suffix _arcmain
  done
}
run_rc TurnOffSinkFaucet 8765 & p1=$!
run_rc CoffeePressButton 8766 & p2=$!
run_rc TurnOffMicrowave 8767 & p3=$!
run_rc CloseSingleDoor 8768 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"

# Separate visual confirmation: ARC is conditioned on RGB+proprioception, not privileged state.
vision_port=8770
pgrep -f "robocasa_vision_bridge.py --port $vision_port" >/dev/null || {
  "$BR" robocasa_vision_bridge.py --port "$vision_port" --data-dir "$VISION_DATA" \
    >"bridge_${vision_port}_arcmain.log" 2>&1 &
  sleep 5
}
for k in "${RATES[@]}"; do
  mode=eval
  [[ $k == 1 ]] && mode=train-eval
  "$RM_PY" -u stage1_vision.py CloseSingleDoor --head arc --mode "$mode" --k "$k" \
    --arms zoh,spline,bspline_eps_raw,tac_fold,tac_fold_satfix \
    --data-dir "$VISION_DATA" --output-dir "$VISION_OUT" \
    --n-train 35 --n-eval 15 --steps 15000 --port "$vision_port"
done
