#!/usr/bin/env bash
set -euo pipefail

cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl LIBERO_SUITE=libero_spatial
export HF_HOME=/tmp/hf_cache HF_LEROBOT_HOME=/tmp/lerobot_cache
PYTHON=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

echo "=== RESUMING ARMS 2 AND 3 ===" >> eval_patched_3arms.log

echo "[2/3] Running tempo_s0..." >> eval_patched_3arms.log
export DEEPONET_HEAD=tempo DEEPONET_STATE_HISTORY_STEPS=1
$PYTHON evaluate_libero_standard.py \
  --model tempo_s0=deeponet=runs/tempo_s0/checkpoints/8300 \
  --out standard_tempo_patched >> eval_patched_3arms.log 2>&1

echo "[3/3] Running til_s0..." >> eval_patched_3arms.log
export DEEPONET_HEAD=til DEEPONET_STATE_HISTORY_STEPS=8
$PYTHON evaluate_libero_standard.py \
  --model til_s0=deeponet=runs/til_s0/checkpoints/8300 \
  --out standard_til_patched >> eval_patched_3arms.log 2>&1

echo "=== ALL 3 ARMS FINISHED ===" >> eval_patched_3arms.log
