#!/usr/bin/env bash
# Phase 2a: train flow@8300 on object / goal / long (sequential), data drive.
set -u
cd '/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH' || exit 1
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
BASE=/media/user/C2FE578FFE577A9D/vla_matched
export CUDA_VISIBLE_DEVICES=0

for S in object goal 10; do
  DS="lerobot/libero_${S}_image"
  OUT="$BASE/flow8300_${S}_s0"
  if [ -f "$OUT/checkpoints/8300/model.safetensors" ]; then
    echo "SKIP $S (checkpoint exists)"
    continue
  fi
  echo "=== training flow8300_${S}_s0 on $DS  ($(date)) ==="
  $PY train.py --head flow --variant baseline --out "$OUT" --dataset "$DS" --seed 0 \
    --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
    --warmup 500 --head_lr 0.0001 --backbone_lr 1e-05 --ema 0.999 \
    --num_workers 8 --ckpt_every 10000 >> "$BASE/flow8300_suites_train.log" 2>&1 \
    && echo "DONE $S ($(date))" || echo "FAIL $S ($(date))"
done
echo ALL_SUITE_TRAINING_DONE
