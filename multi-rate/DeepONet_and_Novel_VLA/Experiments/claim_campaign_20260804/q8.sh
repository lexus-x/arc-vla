#!/bin/bash
# OPEN LOOP -- no replanning of any kind.
#
# The chunk is ceil(HORIZON_S * 20) = 50 steps, so --replan_steps 50 re-queries the policy only
# when the previous chunk is fully consumed. Nothing is discarded, nothing is refreshed early.
#
# Why this protocol: the original authors' eval used replan=5 (v2 85.0/87.0/58.5, flow
# 79.5/87.5/66.5); the q6 grid used replan=10 (v2 27.3/28.7/20.3, flow 80.7/89.7/65.0). v2 moved
# 58 pp, flow moved ~1 pp. Frequent replanning is therefore a crutch that flatters one model and
# not the other. Open loop removes it for BOTH and asks the honest question: whose 50-step chunk
# is actually good?
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q8] $(date +%H:%M) $*" >> final8.log; }
for i in $(seq 1 900); do pgrep -f "[p]ython.*evaluate_multirate_honest" >/dev/null || break; sleep 30; done
log "GPU clear -- open-loop run, replan_steps=50 (= full chunk)"

batch () {
  pids=(); names=(); suites=()
  while [ $# -gt 0 ]; do
    S="$1"; A="$2"; shift 2
    O="ol_${A}_out"
    [ -d "$O" ] && { log "skip $A"; continue; }
    $PY evaluate_multirate_honest.py --arms "$A" --trials_per_task 30 --suite "$S" \
        --replan_steps 50 --out "$O" > "ol_${A}.log" 2>&1 &
    pids+=($!); names+=("$A"); suites+=("$S")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}"; log "${names[$i]} exit=$?"; done
  for i in "${!names[@]}"; do
    A="${names[$i]}"; S="${suites[$i]}"
    F="ol_${A}_out/multirate_${S}.json"
    [ "$S" = "libero_spatial" ] && F="ol_${A}_out/multirate_honest.json"
    [ -s "$F" ] && log "OK json $A" || log "MISSING json $A <-- rerun"
  done
  log "=== batch done ==="
}

batch libero_spatial v2_spatial_native_20env  libero_spatial flow30_spatial_native_20env \
      libero_10      v2_long_native_20env     libero_10      flow30_long_native_20env
batch libero_object  v2_object_native_20env   libero_object  flow30_object_native_20env
log "Q8 DONE"
