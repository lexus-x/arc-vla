#!/bin/bash
# Wait for the `long` training to finish, then run the 40 Hz evals for all three
# per-suite asrc checkpoints. Gate-first, at n=50.
#
# Concurrency: 3 at a time, staggered 30 s. Evals are CPU/EGL-bound, not VRAM-bound, and
# 9 concurrent MuJoCo EGL contexts previously killed 2 jobs at init (silently, writing no
# JSON). 3 is well inside that limit. This is NOT the training constraint -- training was
# limited by stage-2 VRAM (50.7 GB), which does not apply here.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1

cat > SUITEEVAL_PREREG.md <<'PREREG'
# Pre-registration -- per-suite 40 Hz evals (written BEFORE suiteeval_*/ exist)

Checkpoints: asrc_{object,goal,long}_s0 @8300, each trained ONLY on its own suite, config
replicated from asrc_s0 (only --dataset/--out differ). Budget-matched at 8300 steps.

Arms per suite, n=50 (10 tasks x 5 trials), init_state_id pinned, per-suite horizon
(object 280 / goal 300 / long 520 steps at 20 Hz; doubled at 40 Hz):
  <m>_native_20env          20 Hz reference AND GATE
  <m>_cadmag_spline_40env   comparator
  <m>_cadmag_folding_40env  ours

Normalization stats follow the CHECKPOINT via MODEL_DATASET (asrc_object ->
libero_object_image etc), not the eval suite. This matters because
rate_integrated_deeponet.py:130 uses action_scale INSIDE the RK4 integration, so wrong stats
change the head's dynamics rather than merely rescaling output. Each arm prints a [stats] line.

## GATE, declared before any result exists
A suite counts ONLY if <m>_native_20env > 0/50 on its own suite. A suite-trained checkpoint
reading 0/50 on the suite it was trained on is a TRAINING FAILURE, not a rate finding, and must
be reported as such. A 0-vs-0 spline-vs-folding cell is VOID, never a tie.

## What this can claim
n=50 is a GATE, not a result. ~10 discordant pairs at best cannot certify equivalence, and this
project has twice seen n=50 estimates reverse under power (spline@40 78%->72%, RAI@40 66%->71%).
Cells that clear the gate get powered to n=300 before anything is claimed. No directional
hypothesis is pre-specified for the new suites; report two-sided.

## Known caveat, stated in advance
Budget-matching at 8300 steps gives each suite a different number of data passes:
object 5.95 epochs, goal 7.66, long ~3.9 (101k frames). A weak `long` result is therefore
ambiguous between "no decoding benefit" and "undertrained", and cannot be attributed without a
second long run at higher budget.
PREREG

# --- wait for training to clear ---
for i in $(seq 1 480); do
  pgrep -f "[t]rain.py" >/dev/null || break
  sleep 60
done
if pgrep -f "[t]rain.py" >/dev/null; then
  echo "[eval] timed out waiting for training" >> evalqueue.log; exit 1
fi
sleep 30
echo "[eval] $(date +%H:%M) training clear, starting evals" >> evalqueue.log

export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

run_arm() {  # suite_key libero_suite arm
  local M=$1 S=$2 A=$3
  local OUT="suiteeval_${M}_${A}_out"
  [ -f "$OUT"/*.json ] 2>/dev/null && return 0
  LIBERO_SUITE=$S nohup $PY evaluate_multirate_honest.py \
    --arms "$A" --trials_per_task 5 --suite "$S" --out "$OUT" \
    > "suiteeval_${A}.log" 2>&1 &
  sleep 30
}

for pair in "object:libero_object" "goal:libero_goal" "long:libero_10"; do
  M=${pair%%:*}; S=${pair##*:}
  if [ ! -d "runs/asrc_${M}_s0/checkpoints/8300" ]; then
    echo "[eval] SKIP $M -- no checkpoint" >> evalqueue.log; continue
  fi
  echo "[eval] $(date +%H:%M) $M" >> evalqueue.log
  run_arm "$M" "$S" "asrc_${M}_native_20env"
  run_arm "$M" "$S" "asrc_${M}_cadmag_spline_40env"
  run_arm "$M" "$S" "asrc_${M}_cadmag_folding_40env"
  wait
done
echo "[eval] $(date) ALL EVALS DONE" >> evalqueue.log
