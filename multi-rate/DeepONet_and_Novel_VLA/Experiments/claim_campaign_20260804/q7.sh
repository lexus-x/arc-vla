#!/bin/bash
# ISOLATION: is v2's collapse caused by replan alone?
#
# Original authors' eval of these exact 30K weights: replan=5, max_steps=520 -> v2 85.0/87.0/58.5,
# flow 79.5/87.5/66.5. The q6 grid used replan=10 -> v2 27.3/28.7/20.3, flow 80.7/89.7/65.0.
# On Long the horizon already matched (520 both), so replan is the only variable left.
#
# This runs the SAME native-20Hz arms at replan=5, changing nothing else. Prediction if replan is
# the cause: v2 recovers toward its original numbers; flow stays put.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q7] $(date +%H:%M) $*" >> final7.log; }
for i in $(seq 1 900); do pgrep -f "[e]valuate_multirate_honest" >/dev/null || break; sleep 30; done
log "GPU clear, starting replan=5 isolation"

run4 () {
  SUITE="$1"; shift
  pids=(); names=()
  for A in "$@"; do
    O="r5_${A}_out"
    [ -d "$O" ] && { log "skip $A"; continue; }
    $PY evaluate_multirate_honest.py --arms "$A" --trials_per_task 30 --suite "$SUITE" \
        --replan_steps 5 --out "$O" > "r5_${A}.log" 2>&1 &
    pids+=($!); names+=("$A")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}"; log "${names[$i]} exit=$?"; done
  for A in "$@"; do
    F="r5_${A}_out/multirate_${SUITE}.json"
    [ "$SUITE" = "libero_spatial" ] && F="r5_${A}_out/multirate_honest.json"
    [ -s "$F" ] && log "OK json $A" || log "MISSING json $A <-- rerun"
  done
  log "=== $SUITE done ==="
}

# Spatial + Long first: Spatial is the biggest claimed v2 win (85.0 vs flow 79.5), Long is the
# clean case where q6's horizon already equalled the original's.
run4 libero_spatial v2_spatial_native_20env flow30_spatial_native_20env
run4 libero_10      v2_long_native_20env    flow30_long_native_20env
run4 libero_object  v2_object_native_20env  flow30_object_native_20env
log "Q7 DONE"
