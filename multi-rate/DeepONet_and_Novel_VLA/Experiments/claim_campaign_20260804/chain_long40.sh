#!/bin/bash
# Wait for the latency probe to release the GPU, then run the two long 40 Hz arms.
# They previously crashed on robosuite's 1000-step horizon (1040 steps needed at 40 Hz);
# that is now patched. Run them SEQUENTIALLY -- 1040-step rollouts are the most expensive
# cells in the campaign and there is nothing to gain from overlapping them.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
for i in $(seq 1 600); do
  pgrep -f "[p]robe_latency" >/dev/null || break
  sleep 60
done
sleep 20
echo "[long40] $(date +%H:%M) probe clear, starting" >> long40.log
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
for A in asrc_long_cadmag_spline_40env asrc_long_cadmag_folding_40env; do
  OUT="suiteeval_long_${A}_out"
  rm -rf "$OUT"
  echo "[long40] $(date +%H:%M) $A" >> long40.log
  LIBERO_SUITE=libero_10 $PY evaluate_multirate_honest.py \
    --arms "$A" --trials_per_task 5 --suite libero_10 --out "$OUT" \
    > "suiteeval_${A}.log" 2>&1
  echo "[long40] $(date +%H:%M) $A rc=$?" >> long40.log
done
echo "[long40] $(date) DONE" >> long40.log
