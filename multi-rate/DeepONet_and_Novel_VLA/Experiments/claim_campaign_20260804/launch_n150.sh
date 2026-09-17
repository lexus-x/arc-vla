#!/usr/bin/env bash
# EART v1 n=150 confirmation + paired spline n=150 (same pinned trials -> McNemar).
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MAGSCALE_K=0.5
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

$PY - <<'EOF' &
import sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_eart_10env"] = ("flow30_spatial", 10, None, "spline+magscale+eart")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_eart_10env",
            "--trials_per_task", "15", "--suite", "libero_spatial",
            "--out", "eart150_rp5_out"]
E.main()
EOF
echo "eart150 pid $!"

$PY - <<'EOF' &
import sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_sm_10env"] = ("flow30_spatial", 10, None, "spline+magscale")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_sm_10env",
            "--trials_per_task", "15", "--suite", "libero_spatial",
            "--out", "spline150_rp5_out"]
E.main()
EOF
echo "spline150 pid $!"
echo N150_LAUNCHED
