#!/bin/bash
# LIBERO-Plus at 40 Hz: ours (folding) vs flow (spline only -- it cannot fold).
# This is the cell where the matched-budget DeepONet advantage is a ROBUSTNESS effect;
# in-distribution it was a tie, so Spatial is not where a win should be expected.
# Waits for q2 and q3 so nothing contends for the GPU.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=6
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q4] $(date +%H:%M) $*" >> final4.log; }

for i in $(seq 1 900); do pgrep -f "[q]2.sh|[q]3.sh" >/dev/null || break; sleep 60; done
log "q2+q3 clear, starting"

# smoke first: 1 task/category, catches wiring faults before committing hours
log "smoke flow_m1 on Plus"
rm -rf /tmp/smoke_plus_flow
$PY evaluate_plus_multirate.py --arms flow_m1_cadmag_spline_40env --per_category 1 \
  --out /tmp/smoke_plus_flow > smoke_plus_flow.log 2>&1
if [ $? -ne 0 ]; then log "SMOKE FAILED - aborting, see smoke_plus_flow.log"; exit 1; fi
log "smoke ok"

for A in flow_m1_cadmag_spline_40env flow_m1_native_20env; do
  O="pow_plus_${A}_out"; [ -d "$O" ] && { log "skip $A"; continue; }
  log "plus n=315 $A"
  $PY evaluate_plus_multirate.py --arms "$A" --per_category 45 --out "$O" \
    > "powplus_${A}.log" 2>&1
  log "$A exit=$?"
done
log "Q4 DONE"
