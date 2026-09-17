#!/usr/bin/env bash
# Auto-launch spatial eval when the Plus eval completes both arms.
CAMP=/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804
RES="$CAMP/plus_matched8300_out/plus_multirate.json"
while true; do
  if [ -f "$RES" ]; then
    DONE=$(python3 -c "import json; d=json.load(open('$RES')); ok=all(d.get(a,{}).get('aggregate') is not None for a in ['asrc_native_20env','flow8300_native_20env']); print('YES' if ok else 'NO')" 2>/dev/null)
    if [ "$DONE" = "YES" ]; then
      echo "PLUS COMPLETE $(date)"
      break
    fi
  fi
  sleep 120
done
cd "$CAMP" || exit 1
nohup bash run_matched8300_spatial.sh > matched8300_spatial_run.log 2>&1 &
echo "SPATIAL_EVAL_LAUNCHED pid $! at $(date)"
