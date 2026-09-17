#!/bin/bash
# Run goal then long SEQUENTIALLY, each only after the GPU is free of a stage-2 run.
#
# Why sequential: stage1 trains the head only (10.46M params, 9.9 GB). Stage2 unfreezes the
# backbone -> 50.7 GB. Two stage-2 runs need ~101 GB against 94.96 GB usable, so the second
# one OOMs the instant it transitions. Concurrency was sized off a stage-1 measurement; that
# was the mistake. Batch size is NOT reduced to fit more runs -- that would break the
# budget/recipe match with asrc_s0, which is the whole point of these checkpoints.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1

wait_for_free_gpu() {
  # wait until no train.py is running (the previous run has fully exited and released VRAM)
  for i in $(seq 1 480); do
    pgrep -f "[t]rain.py" >/dev/null || { sleep 30; return 0; }
    sleep 60
  done
  echo "[queue] timed out waiting for a free GPU" >> queue.log
  return 1
}

for R in goal long; do
  case $R in
    goal) DS=lerobot/libero_goal_image ;;
    long) DS=lerobot/libero_10_image ;;
  esac
  if [ -d "runs/asrc_${R}_s0/checkpoints/8300" ]; then
    echo "[queue] $R already complete, skipping" >> queue.log
    continue
  fi
  wait_for_free_gpu || exit 1
  # start clean: the dead run left a partial 1650 ckpt and train.py cannot resume from it
  rm -rf "runs/asrc_${R}_s0"
  echo "[queue] $(date +%H:%M) starting $R on $DS" >> queue.log
  ./train_suite.sh "$DS" "runs/asrc_${R}_s0" > "train_${R}.log" 2>&1
  echo "[queue] $(date +%H:%M) $R exited rc=$?" >> queue.log
done
echo "[queue] $(date) ALL TRAININGS DONE" >> queue.log
