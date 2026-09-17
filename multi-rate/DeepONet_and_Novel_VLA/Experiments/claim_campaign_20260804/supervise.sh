#!/bin/bash
# Relaunch any 4-suite arm that has no result JSON and no live process.
# EGL context creation fails under concurrent init, so retries are staggered.
# Max 4 attempts per arm, then give up and record it.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

for round in $(seq 1 40); do
  sleep 300
  alldone=1
  for SUITE in libero_object libero_goal libero_10; do
    for ARM in asrc_native_20env asrc_cadmag_spline_40env asrc_cadmag_folding_40env; do
      OUT="foursuite_${SUITE}_${ARM}_out"
      LOG="foursuite_${SUITE}_${ARM}.log"
      J="$OUT/multirate_honest.json"
      # complete?
      if [ -f "$J" ] && grep -q '"aggregate"' "$J" 2>/dev/null \
         && ! grep -q '"aggregate": null' "$J" 2>/dev/null; then continue; fi
      alldone=0
      # live?
      if pgrep -f "[-]-out $OUT" >/dev/null 2>&1; then continue; fi
      N=$(cat "${OUT}.attempts" 2>/dev/null || echo 0)
      if [ "$N" -ge 4 ]; then continue; fi
      echo $((N+1)) > "${OUT}.attempts"
      echo "[sup $(date +%H:%M)] relaunch #$((N+1)) $SUITE/$ARM" >> supervise.log
      LIBERO_SUITE=$SUITE setsid nohup $PY evaluate_multirate_honest.py \
        --arms "$ARM" --trials_per_task 5 --suite "$SUITE" --out "$OUT" \
        > "$LOG" 2>&1 < /dev/null &
      sleep 30
    done
  done
  [ "$alldone" = "1" ] && { echo "[sup $(date)] ALL COMPLETE" >> supervise.log; exit 0; }
done
echo "[sup $(date)] supervisor exiting after 40 rounds" >> supervise.log
