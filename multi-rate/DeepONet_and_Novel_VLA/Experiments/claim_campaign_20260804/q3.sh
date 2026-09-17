#!/bin/bash
# Head-to-head vs FLOW MATCHING at a matched 8,300-step budget. See FLOW_PREREG.md.
# Waits for q2 (standard protocol + Plus n=315) so nothing contends.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=6
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q3] $(date +%H:%M) $*" >> final3.log; }

for i in $(seq 1 480); do pgrep -f "[q]2.sh" >/dev/null || break; sleep 60; done
log "q2 clear, starting"

# H1 -- 20 Hz head quality, matched budget. Never run before.
# H2 -- 40 Hz deployed: ours folds, flow can only resample.
for A in asrc_native_20env flow_m1_native_20env don_v2_native_20env flow_m1_cadmag_spline_40env; do
  O="h2h_${A}_out"; [ -d "$O" ] && { log "skip $A (exists)"; continue; }
  log "n=300 $A"
  LIBERO_SUITE=libero_spatial $PY evaluate_multirate_honest.py --arms "$A" \
    --trials_per_task 30 --suite libero_spatial --out "$O" > "h2h_${A}.log" 2>&1
  log "$A exit=$? $(grep -oE '[0-9.]+%$' "h2h_${A}.log" | tail -1)"
done
log "Q3 DONE"
