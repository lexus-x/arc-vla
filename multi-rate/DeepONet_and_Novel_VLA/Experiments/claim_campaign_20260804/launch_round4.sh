#!/usr/bin/env bash
# Round 4 (2026-08-21 ~18:05): close 40 Hz the same way + flow@10Hz SOTA baselines.
# A) 40Hz-targeted consistency fine-tune from asrc_s0 (2000 steps)
# B) flow30 @ 10 Hz cubic-spline downsample, replan 5 (0.5 s) — SOTA baseline, n=50
# C) flow30 @ 10 Hz cubic-spline downsample, replan 10 (1.0 s) — SOTA baseline, n=50
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

nohup $PY train.py \
  --head deeponet --variant baseline --deeponet_head asrc \
  --base_ckpt runs/asrc_s0/checkpoints/8300 \
  --dataset lerobot/libero_spatial_image --out runs/asrc40hz_ft_s0 \
  --deeponet_p 256 --deeponet_fourier 6 --deeponet_blocks 3 --deeponet_queries 8 \
  --consistency_rates 40 --rate_consistency_weight 0.2 \
  --trunk_bandlimit --state_history_steps 8 --warmup 200 \
  --stage1_steps 2000 --stage2_steps 0 --stage1_batch 48 \
  --head_lr 1e-4 --ema 0.999 --num_workers 8 --seed 0 \
  --epoch_steps 200 --ckpt_every 2000 \
  > train_asrc40hz_ft_s0.log 2>&1 &
echo "ft40 pid $!"

# flow30 @ 10 Hz spline arms (flow has no ODE; spline downsample with displacement
# preserving rescale is its best possible off-native decode).
$PY - <<'EOF' &
import os, sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_spline_10env"] = ("flow30_spatial", 10, None, "spline")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_spline_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "flow10_rp5_out"]
E.main()
EOF
echo "flow10_rp5 pid $!"

$PY - <<'EOF' &
import os, sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_spline_10env"] = ("flow30_spatial", 10, None, "spline")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_spline_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--replan_steps", "10", "--out", "flow10_rp10_out"]
E.main()
EOF
echo "flow10_rp10 pid $!"

sleep 3
pgrep -af "train.py --head deeponet --variant baseline --deeponet_head asrc" | wc -l
echo ROUND4_LAUNCHED
