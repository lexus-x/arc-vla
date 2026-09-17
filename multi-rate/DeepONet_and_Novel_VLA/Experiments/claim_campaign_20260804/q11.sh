#!/bin/bash
# PRE-REGISTERED: does folding's exact rate-invariance pay off OPEN LOOP at 40 Hz?
#
# Mechanism: folding emits a_k = integral of v over step k's slice, so sum_k a_k = s(T) exactly
# at any rate (measured: 0.0002% error vs spline's 2.25% at 40 Hz). Every previous
# folding-vs-spline test was CLOSED loop (replan 1-10), where the controller re-observes every
# few steps and wipes out that error -- all four came back ties (p = 0.23 / 0.50 / 1.00 / 0.86).
# Open loop removes the correction: 100 steps at 40 Hz executed blind, so a conservation error
# compounds instead of being erased.
#
# PREDICTION: folding > spline, open loop, 40 Hz. Same checkpoint, decoder is the only variable.
# KILL RULE: if the paired delta is not positive and p < 0.05, the exactness advantage does not
# convert to task success anywhere, and this line is closed for good.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q11] $(date +%H:%M) $*" >> final11.log; }
for i in $(seq 1 900); do pgrep -f '[p]ython.*evaluate_multirate_honest' >/dev/null || break; sleep 20; done
log 'GPU clear -- 40 Hz OPEN LOOP, folding vs spline, same asrc ckpt'
# chunk at 40 Hz is ceil(2.5*40) = 100 steps -> replan_steps 100 = fully open loop
pids=(); names=()
for A in asrc_cadmag_folding_40env asrc_cadmag_spline_40env; do
  $PY evaluate_multirate_honest.py --arms $A --trials_per_task 30 --suite libero_spatial       --replan_steps 100 --out olf_${A}_out > olf_${A}.log 2>&1 &
  pids+=($!); names+=($A)
done
for i in ${!pids[@]}; do wait ${pids[$i]}; log "${names[$i]} exit=$?"; done
log 'Q11 DONE'
