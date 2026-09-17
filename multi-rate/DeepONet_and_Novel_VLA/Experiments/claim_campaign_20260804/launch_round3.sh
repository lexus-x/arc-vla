#!/usr/bin/env bash
# Round 3 (2026-08-21 ~16:40): paired controls + fine-tuned head evals.
# A) native 20 Hz @ replan 20 (1.0 s physical — paired control for rp10 @ 10 Hz)
# B) asrc10hz_ft @ 10 Hz, k=0.5, replan 5 (physical-matched)
# C) asrc10hz_ft @ 10 Hz, k=0.5, replan 10 (replan-count-matched)
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

# Wait for the ft training checkpoint if it is still running.
for i in $(seq 1 60); do
  [ -d runs/asrc10hz_ft_s0/checkpoints/2000 ] && break
  sleep 30
done

$PY evaluate_multirate_honest.py \
  --arms asrc_native_20env --trials_per_task 5 --suite libero_spatial \
  --replan_steps 20 --out rp20_native_20env_out > rp20_native_20env.log 2>&1 &
echo "rp20 pid $!"

FT_CKPT="/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/runs/asrc10hz_ft_s0/checkpoints/2000"

$PY - "$FT_CKPT" <<'EOF' &
import sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import os
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
os.environ["MAGSCALE_K"] = "0.5"
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_mag_folding_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "ft10_rp5_out"]
E.main()
EOF
echo "ft_rp5 pid $!"

$PY - "$FT_CKPT" <<'EOF' &
import sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import os
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
os.environ["MAGSCALE_K"] = "0.5"
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_mag_folding_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--replan_steps", "10", "--out", "ft10_rp10_out"]
E.main()
EOF
echo "ft_rp10 pid $!"

sleep 3
pgrep -af "evaluate_multirate_honest" | head -8
echo ROUND3_LAUNCHED
