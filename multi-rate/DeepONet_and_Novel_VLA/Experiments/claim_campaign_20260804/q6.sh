#!/bin/bash
# Matched-budget 30K grid: {DeepONet-v2, flow} x {native 20 Hz, cadence+magscale+spline 40 Hz}
# across Spatial / Object / Long. n=300 per arm (10 tasks x 30 trials), paired by task+trial.
#
# 4-way parallel is the EGL ceiling: 9 concurrent MuJoCo contexts previously lost 2 jobs to
# EGLError while writing NO json, so every batch asserts its output files exist before moving on.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
TRIALS=30
log(){ echo "[q6] $(date +%H:%M) $*" >> final6.log; }

# wait for anything still holding the GPU
for i in $(seq 1 900); do pgrep -f "[e]valuate_plus_multirate|[e]valuate_libero_standard" >/dev/null || break; sleep 30; done
log "GPU clear, starting grid (trials_per_task=$TRIALS)"

run_suite () {
  SUITE="$1"; TAG="$2"; shift 2
  log "=== $SUITE : $* ==="
  pids=(); names=()
  for A in "$@"; do
    O="grid_${A}_out"
    if [ -d "$O" ]; then log "skip $A (exists)"; continue; fi
    $PY evaluate_multirate_honest.py --arms "$A" --trials_per_task $TRIALS \
        --suite "$SUITE" --out "$O" > "grid_${A}.log" 2>&1 &
    pids+=($!); names+=("$A")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; log "${names[$i]} exit=$?"
  done
  # EGL deaths are silent -- verify each arm actually produced its json
  for A in "$@"; do
    F="grid_${A}_out/multirate_${SUITE}.json"
    [ "$SUITE" = "libero_spatial" ] && F="grid_${A}_out/multirate_honest.json"
    if [ -s "$F" ]; then log "OK  json $A"; else log "MISSING json $A  <-- rerun needed"; fi
  done
  log "=== $SUITE done ==="
}

run_suite libero_spatial SP v2_spatial_native_20env v2_spatial_cadmag_spline_40env \
                             flow30_spatial_native_20env flow30_spatial_cadmag_spline_40env
run_suite libero_object  OB v2_object_native_20env v2_object_cadmag_spline_40env \
                             flow30_object_native_20env flow30_object_cadmag_spline_40env
run_suite libero_10      LG v2_long_native_20env v2_long_cadmag_spline_40env \
                             flow30_long_native_20env flow30_long_cadmag_spline_40env
log "Q6 DONE"
