#!/bin/bash
# Runs baseline eval -> RL pilot training -> RL eval for one LIBERO suite.
set -e
SUITE=$1
CKPT_DIR=$2
cd /home/user/Desktop/multi-rate/vla-rft

echo "[$SUITE] === baseline eval ==="
conda run -n vla_smolvla_libero python baseline_eval_suite.py \
  --suite "$SUITE" --ckpt "$CKPT_DIR" --trials_per_task 10 \
  --out "baseline_results/${SUITE}_seed0"

echo "[$SUITE] === RL GRPO pilot ==="
conda run -n vla_smolvla_libero python train_grpo_suite.py \
  --suite "$SUITE" --ckpt "$CKPT_DIR" \
  --group_size 6 --n_groups_per_update 2 --n_updates 8 \
  --out_ckpt "rl_checkpoint_${SUITE}" \
  --log_json "logs/train_progress_${SUITE}.json"

echo "[$SUITE] === RL eval ==="
conda run -n vla_smolvla_libero python rl_eval_suite.py \
  --suite "$SUITE" --ckpt "rl_checkpoint_${SUITE}" \
  --baseline_summary "baseline_results/${SUITE}_seed0/summary.json" \
  --trials_per_task 10 \
  --out "baseline_results/rl_${SUITE}_seed0"

echo "[$SUITE] === DONE ==="
