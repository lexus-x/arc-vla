#!/usr/bin/env bash
# Overnight combo (2026-08-21 ~23:15): asrc ft with consistency_rates=10,40 (weight 0.2,
# 4000 head-only steps) from asrc_s0; then eval 10Hz rp5/rp10 + 40Hz (n=50 each).
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

$PY train.py \
  --head deeponet --variant baseline --deeponet_head asrc \
  --base_ckpt runs/asrc_s0/checkpoints/8300 \
  --dataset lerobot/libero_spatial_image --out runs/asrc_combo_ft_s0 \
  --deeponet_p 256 --deeponet_fourier 6 --deeponet_blocks 3 --deeponet_queries 8 \
  --consistency_rates 10,40 --rate_consistency_weight 0.2 \
  --trunk_bandlimit --state_history_steps 8 --warmup 200 \
  --stage1_steps 4000 --stage2_steps 0 --stage1_batch 48 \
  --head_lr 1e-4 --ema 0.999 --num_workers 8 --seed 0 \
  --epoch_steps 200 --ckpt_every 4000 \
  > train_asrc_combo_ft_s0.log 2>&1
echo "combo ft done rc=$?"

FT="$PWD/runs/asrc_combo_ft_s0/checkpoints/4000"
[ -d "$FT" ] || { echo "NO FT CKPT"; exit 1; }

MAGSCALE_K=0.5 $PY - "$FT" <<'EOF' &
import os, sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
os.environ["MAGSCALE_K"] = "0.5"
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_mag_folding_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "combo_10_rp5_out"]
E.main()
EOF

MAGSCALE_K=0.5 $PY - "$FT" <<'EOF' &
import os, sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
os.environ["MAGSCALE_K"] = "0.5"
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_mag_folding_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--replan_steps", "10", "--out", "combo_10_rp10_out"]
E.main()
EOF

$PY - "$FT" <<'EOF' &
import sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_cadmag_folding_40env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "combo_40_out"]
E.main()
EOF
wait
echo COMBO_ALL_DONE
