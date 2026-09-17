#!/bin/bash
# Parallel-worker launcher for multi_task_campaign.py, same pattern as this project's
# run_campaign_worker.sh: WORKER_COUNT workers, each partitioning the fixed
# (task_id x arm) grid deterministically via --worker_id/--worker_count, sharing the GPU.
# 4 workers matches the precedent validated in the prior 16-run collapse-aware-GRPO
# campaign (~2.4GB VRAM / ~33% util per job on this same GPU/model class -> plenty of
# headroom for 4 concurrent). Detached via setsid+nohup so it survives independently.
WORKER_COUNT=${1:-4}
cd /home/user/Desktop/multi-rate/vla-rft || exit 1
mkdir -p logs

echo "{\"event\":\"CAMPAIGN_LAUNCH\",\"workers\":$WORKER_COUNT,\"ts\":\"$(date -Is)\"}" >> logs/multi_task_campaign_manifest.jsonl

for ((w=0; w<WORKER_COUNT; w++)); do
  setsid nohup /home/user/anaconda3/envs/vla_smolvla_libero/bin/python3 multi_task_campaign.py \
    --worker_id "$w" --worker_count "$WORKER_COUNT" \
    > "logs/multi_task_campaign_worker${w}.log" 2>&1 < /dev/null &
  disown
  echo "launched worker $w pid $!"
done
