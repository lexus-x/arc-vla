#!/usr/bin/env bash
# Complete LIBERO suite sweep, queued behind the running LIBERO-Plus job.
# Concurrent MuJoCo evals saturate CPU and slow each other 3-4x, so we wait.
#
# Already done, NOT redone (the harness skips completed arms in place):
#   libero_spatial : til + asrc, 20/30/40 Hz, all transforms   -> multirate_honest.json
#   LIBERO-Plus    : asrc ours/spline/native @30-20 Hz          -> plus_multirate.json
# Left out deliberately: MetaWorld. Its action space is (4,) vs the policy's 7, obs is a
# 39-dim state vector with no cameras, and the robot/tasks were never trained on. Every arm
# would score 0% and nothing about multi-rate would be tested.
set -u
C=/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
ARMS=asrc_cadmag_folding_30env,asrc_cadmag_spline_30env,asrc_native_20env
cd "$C" || exit 1

echo "[queue] waiting for LIBERO-Plus to finish at $(date -Is)"
while pgrep -f "[b]in/python evaluate_plus_multirate" >/dev/null; do sleep 60; done
echo "[queue] Plus done at $(date -Is)"

for SUITE in libero_object libero_goal libero_10; do
  echo "=============================================================="
  echo "[suite] $SUITE starting at $(date -Is)"
  echo "  NOTE: checkpoints trained on libero_spatial only -> this is CROSS-SUITE OOD."
  echo "  Absolute numbers will be low; the ours-vs-spline contrast within the suite is"
  echo "  still valid because both arms use the same weights and the same tasks."
  echo "=============================================================="
  $PY evaluate_multirate_honest.py \
      --arms "$ARMS" --trials_per_task 5 --suite "$SUITE" \
      --out multirate_honest_out > "suite_${SUITE}.log" 2>&1
  echo "[suite] $SUITE exit=$? at $(date -Is)"
  grep -E "^--> |diagnostics" "suite_${SUITE}.log" | tail -8
done

echo "[queue] ALL SUITES COMPLETE at $(date -Is)"
