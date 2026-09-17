#!/bin/bash
# Full grid: {RoboMimic, ManiSkill(PushT/PickCube), RoboCasa} x {DP, FM}, plus bspline re-eval
# on the already-trained DP checkpoints from dp_tacfold_2026-09-06. Protocol: PREREG.md.
# Skip-if-done via result_<tag>.json. RoboCasa needs the bridge process (robocasa_uv) running
# first -- started and torn down by this script.
set -u
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
MS3=/home/user/anaconda3/envs/ms3/bin/python
RM=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
GR00T=/home/user/anaconda3/envs/gr00t/bin/python
BRIDGE=/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python

run() {
  py=$1; task=$2; policy=$3; extra=${4:-}
  tag="${policy}_${task}"
  [ -f "result_${tag}.json" ] && { echo "[skip] $tag done $(date +%H:%M:%S)"; return; }
  echo "[start] $tag $(date +%H:%M:%S)"
  $py harness.py "$task" --policy "$policy" $extra > "log_${tag}.txt" 2>&1
  rc=$?
  echo "[end] $tag rc=$rc $(date +%H:%M:%S)"
}

# ---- Stream A: ManiSkill (ms3 env) -- bspline re-eval on existing DP ckpt, then new FM train
( run $MS3 PushT-v1    dp; run $MS3 PushT-v1    fm
  run $MS3 PickCube-v1 dp; run $MS3 PickCube-v1 fm ) &
PID_A=$!

# ---- Stream B: RoboMimic (vla_smolvla_libero env) -- same pattern
( run $RM lift   dp; run $RM lift   fm
  run $RM can    dp; run $RM can    fm
  run $RM square dp; run $RM square fm ) &
PID_B=$!

# ---- Stream C: RoboCasa -- bridge (robocasa_uv, CPU sim) + eval driver (gr00t env, GPU policy)
if ! nc -z 127.0.0.1 8765 2>/dev/null; then
  nohup $BRIDGE robocasa_bridge.py --port 8765 > bridge.log 2>&1 &
  BRIDGE_PID=$!
  echo "[bridge] started pid=$BRIDGE_PID, waiting for it to listen..."
  for i in $(seq 1 30); do nc -z 127.0.0.1 8765 2>/dev/null && break; sleep 1; done
else
  BRIDGE_PID=""
  echo "[bridge] already listening on 8765"
fi
( run $GR00T RC-OpenDrawer         dp "--n_train 39 --n_eval 15 --steps 15000"
  run $GR00T RC-OpenDrawer         fm "--n_train 39 --n_eval 15 --steps 15000"
  run $GR00T RC-PnPCounterToStove  dp "--n_train 39 --n_eval 15 --steps 15000"
  run $GR00T RC-PnPCounterToStove  fm "--n_train 39 --n_eval 15 --steps 15000" ) &
PID_C=$!

wait $PID_A $PID_B $PID_C
echo "[campaign] ALL DONE $(date +%H:%M:%S)"
if [ -n "${BRIDGE_PID:-}" ]; then kill "$BRIDGE_PID" 2>/dev/null; echo "[bridge] stopped"; fi
