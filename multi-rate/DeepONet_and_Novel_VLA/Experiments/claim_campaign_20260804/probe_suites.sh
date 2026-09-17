#!/usr/bin/env bash
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
for d in libero_object_image libero_goal_image libero_10_image; do
  p="$HOME/.cache/huggingface/lerobot/hub/datasets--lerobot--$d"
  if [ -d "$p" ]; then
    echo "$d: cached ($(du -sh "$p" 2>/dev/null | cut -f1))"
  else
    echo "$d: NOT cached"
  fi
done
echo "--- per-suite asrc native arm names ---"
grep -n 'asrc_object_native\|asrc_goal_native\|asrc_long_native' evaluate_multirate_honest.py
echo "--- SUITE_WALLCLOCK ---"
grep -n -A6 'SUITE_WALLCLOCK' evaluate_multirate_honest.py | head -10
