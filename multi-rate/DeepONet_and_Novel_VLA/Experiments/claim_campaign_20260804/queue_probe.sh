#!/bin/bash
# Run the latency probe only after ALL training and evals are done. Zero contention.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
for i in $(seq 1 720); do
  if ! pgrep -f '[t]rain.py' >/dev/null && ! pgrep -f '[e]valuate_multirate_honest' >/dev/null; then
    sleep 60
    pgrep -f '[t]rain.py' >/dev/null || pgrep -f '[e]valuate_multirate_honest' >/dev/null || break
  fi
  sleep 60
done
echo "[probe] $(date +%H:%M) starting latency probe" >> probequeue.log
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
/home/user/anaconda3/envs/vla_smolvla_libero/bin/python probe_latency.py > probe_latency.log 2>&1
echo "[probe] $(date +%H:%M) done rc=$?" >> probequeue.log
