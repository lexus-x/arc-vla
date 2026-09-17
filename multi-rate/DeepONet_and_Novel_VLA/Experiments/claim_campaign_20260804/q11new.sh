#!/bin/bash
# PRE-REGISTERED: does folding's exact rate-invariance pay off OPEN LOOP at 40 Hz?
#
# Mechanism. Folding emits a_k = integral of v over step k's own slice of time, so
# sum_k a_k = s(T) exactly at any rate. Measured on real captured contexts: 0.0002% error vs
# cubic spline's 2.25% at 40 Hz, and 0.021% vs 3.284% when downsampling to 10 Hz.
#
# Why it has never shown up in success rate. All four prior folding-vs-spline tests ran CLOSED
# loop (replan 1-10): the controller re-observes every few steps and wipes the displacement error
# out before it can matter. All four were ties -- p = 0.23 / 0.50 / 1.00 / 0.86.
#
# This run removes the correction. At 40 Hz the chunk is ceil(2.5*40) = 100 steps, so
# --replan_steps 100 executes the entire chunk blind. A conservation error now compounds over 100
# steps instead of being erased. This is the only regime where the proven property can convert
# into task success.
#
# PREDICTION: folding > spline. Same checkpoint (asrc_s0@8300), same seeds, same magnitude and
# cadence fixes -- the decoder is the ONLY variable.
# KILL RULE: if the paired delta is not positive with p < 0.05, folding's exactness does not
# convert to task success in any regime we can construct, and the line closes for good.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q11] $(date +%H:%M) $*" >> final11.log; }
log "starting -- 40 Hz OPEN LOOP (replan_steps=100), folding vs spline, asrc_s0@8300"
pids=(); names=()
for A in asrc_cadmag_folding_40env asrc_cadmag_spline_40env; do
  rm -rf "olf_${A}_out"
  $PY evaluate_multirate_honest.py --arms "$A" --trials_per_task 30 --suite libero_spatial \
      --replan_steps 100 --out "olf_${A}_out" > "olf_${A}.log" 2>&1 &
  pids+=($!); names+=("$A")
done
for i in "${!pids[@]}"; do wait "${pids[$i]}"; log "${names[$i]} exit=$?"; done
for A in asrc_cadmag_folding_40env asrc_cadmag_spline_40env; do
  [ -s "olf_${A}_out/multirate_honest.json" ] && log "OK json $A" || log "MISSING json $A"
done
log "Q11 DONE"
