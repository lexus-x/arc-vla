#!/bin/bash
# Redo the two asrc Plus n=315 arms that died on the action-stats shape assertion.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=6
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q5] $(date +%H:%M) $*" >> final5.log; }
for i in $(seq 1 900); do pgrep -f "[q]3.sh|[q]4.sh" >/dev/null || break; sleep 60; done
log "q3+q4 clear, starting"
for A in asrc_cadmag_spline_40env asrc_cadmag_folding_40env; do
  O="pow_plus_${A}_out"; [ -d "$O" ] && { log "skip $A"; continue; }
  log "plus n=315 $A"
  $PY evaluate_plus_multirate.py --arms "$A" --per_category 45 --out "$O" \
    > "powplus_${A}.log" 2>&1
  log "$A exit=$?"
done
log "Q5 DONE"
