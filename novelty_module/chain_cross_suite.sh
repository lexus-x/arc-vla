#!/bin/bash
# Chain: wait for the DeepONet pilot to finish, then run flow pilots on the
# remaining 3 LIBERO suites sequentially (single-GPU contention avoidance).
DEEP_PID=$1
echo "[chain] waiting for deeponet pilot PID $DEEP_PID ..."
while kill -0 "$DEEP_PID" 2>/dev/null; do sleep 60; done
echo "[chain] deeponet done at $(date). Starting cross-suite flow pilots."
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
cd /home/user/Desktop/novelty_module
for SUITE in libero_goal libero_object libero_10; do
  echo "[chain] $SUITE start $(date)"
  $PY eval_canon_pilot.py \
      --head flow \
      --ckpt /media/user/C2FE578FFE577A9D/vla_matched/flow8300_${SUITE#libero_}_s0/checkpoints/8300 \
      --suite $SUITE \
      --out /home/user/Desktop/novelty_module/results/pilot_flow_${SUITE#libero_}.json \
      > pilot_flow_${SUITE#libero_}.log 2>&1
  echo "[chain] $SUITE exit=$? at $(date)"
done
echo "[chain] ALL DONE $(date)"