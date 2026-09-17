#!/bin/bash
# FULL replan=5 GRID -- v2 vs flow, native 20 Hz, all three suites, n=300 paired.
#
# Protocol matched to the original authors' eval exactly: replan=5, max_steps=520 for EVERY suite
# (--wallclock 26.0 at 20 Hz = 520 steps). That makes these numbers directly comparable to their
# published v2 85.0/87.0/58.5 and flow 79.5/87.5/66.5, which is the validation this campaign has
# been missing.
#
# State history is now head-keyed (v2 -> 1, ti/til/asrc -> 8). Calibration after that fix returned
# 83.5% on Spatial against their 85.0%.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q10] $(date +%H:%M) $*" >> final10.log; }
for i in $(seq 1 900); do pgrep -f "[p]ython.*evaluate_multirate_honest" >/dev/null || break; sleep 20; done
log "GPU clear -- replan=5 grid, 520 steps all suites, n=300"

batch () {
  pids=(); names=(); suites=()
  while [ $# -gt 0 ]; do
    S="$1"; A="$2"; shift 2
    O="r5g_${A}_out"
    [ -d "$O" ] && { log "skip $A"; continue; }
    $PY evaluate_multirate_honest.py --arms "$A" --trials_per_task 30 --suite "$S" \
        --replan_steps 5 --wallclock 26.0 --out "$O" > "r5g_${A}.log" 2>&1 &
    pids+=($!); names+=("$A"); suites+=("$S")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}"; log "${names[$i]} exit=$?"; done
  for i in "${!names[@]}"; do
    A="${names[$i]}"; S="${suites[$i]}"
    F="r5g_${A}_out/multirate_${S}.json"
    [ "$S" = "libero_spatial" ] && F="r5g_${A}_out/multirate_honest.json"
    [ -s "$F" ] && log "OK json $A" || log "MISSING json $A <-- rerun"
  done
  log "=== batch done ==="
}

batch libero_spatial v2_spatial_native_20env  libero_spatial flow30_spatial_native_20env \
      libero_object  v2_object_native_20env   libero_object  flow30_object_native_20env
batch libero_10      v2_long_native_20env     libero_10      flow30_long_native_20env
log "Q10 DONE"
