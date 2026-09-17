#!/bin/bash
# Moderate-scope collapse-aware-GRPO campaign: 4 suites x 2 conditions
# (vanilla vs fixed: full_task_coverage + dynamic_sampling + entropy_coef)
# x 2 seeds = 16 train+eval runs. Matched compute (n_updates=24) in both
# conditions so the comparison isn't confounded by training length.
# Continues past a single failed run (logs it in the manifest) rather than
# aborting the whole multi-day campaign.
cd /home/user/Desktop/multi-rate/vla-rft || exit 1

declare -A CKPTS=(
  [libero_spatial]="/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300"
  [libero_object]="/media/user/C2FE578FFE577A9D/vla_matched/flow8300_object_s0/checkpoints/8300"
  [libero_goal]="/media/user/C2FE578FFE577A9D/vla_matched/flow8300_goal_s0/checkpoints/8300"
  [libero_10]="/media/user/C2FE578FFE577A9D/vla_matched/flow8300_10_s0/checkpoints/8300"
)
declare -A BASELINE_SUMMARY=(
  [libero_spatial]="baseline_results/libero_spatial_seed0_v2/summary.json"
  [libero_object]="baseline_results/libero_object_seed0_v2/summary.json"
  [libero_goal]="baseline_results/libero_goal_seed0_v2/summary.json"
  [libero_10]="baseline_results/libero_10_seed0_v2/summary.json"
)

MANIFEST=logs/campaign_manifest.jsonl
mkdir -p logs baseline_results
echo "{\"event\":\"CAMPAIGN_START\",\"ts\":\"$(date -Is)\"}" >> "$MANIFEST"

for suite in libero_spatial libero_object libero_goal libero_10; do
  for cond in vanilla fixed; do
    for seed in 0 1; do
      TAG="${suite}_${cond}_s${seed}"
      CKPT_OUT="rl_checkpoint_${TAG}"
      LOG_JSON="logs/train_progress_${TAG}.json"
      TRAIN_LOG="logs/train_grpo_${TAG}.log"
      EVAL_OUT="baseline_results/rl_${TAG}"
      EVAL_LOG="logs/rl_eval_${TAG}.log"

      if [ -f "${EVAL_OUT}/rl_vs_baseline.json" ]; then
        echo "{\"event\":\"SKIP_DONE\",\"tag\":\"$TAG\",\"ts\":\"$(date -Is)\"}" >> "$MANIFEST"
        continue
      fi

      if [ "$cond" = "vanilla" ]; then
        FLAGS=()
      else
        FLAGS=(--full_task_coverage --dynamic_sampling --max_resample 2 --entropy_coef 0.01)
      fi

      echo "{\"event\":\"TRAIN_START\",\"tag\":\"$TAG\",\"ts\":\"$(date -Is)\"}" >> "$MANIFEST"
      conda run -n vla_smolvla_libero python train_grpo_suite.py \
        --suite "$suite" --ckpt "${CKPTS[$suite]}" --seed "$seed" \
        --group_size 6 --n_groups_per_update 2 --n_updates 24 \
        "${FLAGS[@]}" \
        --out_ckpt "$CKPT_OUT" --log_json "$LOG_JSON" \
        > "$TRAIN_LOG" 2>&1
      TRAIN_RC=$?
      echo "{\"event\":\"TRAIN_END\",\"tag\":\"$TAG\",\"rc\":$TRAIN_RC,\"ts\":\"$(date -Is)\"}" >> "$MANIFEST"

      if [ $TRAIN_RC -ne 0 ]; then
        echo "{\"event\":\"SKIP_EVAL_TRAIN_FAILED\",\"tag\":\"$TAG\",\"ts\":\"$(date -Is)\"}" >> "$MANIFEST"
        continue
      fi

      echo "{\"event\":\"EVAL_START\",\"tag\":\"$TAG\",\"ts\":\"$(date -Is)\"}" >> "$MANIFEST"
      conda run -n vla_smolvla_libero python rl_eval_suite.py \
        --suite "$suite" --ckpt "$CKPT_OUT" \
        --baseline_summary "${BASELINE_SUMMARY[$suite]}" \
        --trials_per_task 10 --out "$EVAL_OUT" \
        > "$EVAL_LOG" 2>&1
      EVAL_RC=$?
      echo "{\"event\":\"EVAL_END\",\"tag\":\"$TAG\",\"rc\":$EVAL_RC,\"ts\":\"$(date -Is)\"}" >> "$MANIFEST"
    done
  done
done
echo "{\"event\":\"CAMPAIGN_COMPLETE\",\"ts\":\"$(date -Is)\"}" >> "$MANIFEST"
