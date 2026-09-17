#!/bin/bash
# Re-run the OPEN-LOOP v2 arms with state history fixed (head-keyed: v2 -> 1).
# Flow arms are NOT re-run: backbone!=deeponet always took the else branch (=1), and flow
# reproduces the original harness within ~1 pp on all three suites. Same seeds, same protocol,
# so ol_flow30_* pairs directly with these.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q9] $(date +%H:%M) $*" >> final9.log; }
for i in $(seq 1 900); do pgrep -f '[c]alib2.sh' >/dev/null || break; sleep 20; done
log 'calib clear -- open-loop v2 re-run, state history fixed'
pids=(); names=(); suites=()
for pair in 'libero_spatial:v2_spatial_native_20env' 'libero_object:v2_object_native_20env' 'libero_10:v2_long_native_20env'; do
  S=${pair%%:*}; A=${pair##*:}
  $PY evaluate_multirate_honest.py --arms $A --trials_per_task 30 --suite $S       --replan_steps 50 --out ol2_${A}_out > ol2_${A}.log 2>&1 &
  pids+=($!); names+=($A); suites+=($S)
done
for i in ${!pids[@]}; do wait ${pids[$i]}; log "${names[$i]} exit=$?"; done
log 'Q9 DONE'
