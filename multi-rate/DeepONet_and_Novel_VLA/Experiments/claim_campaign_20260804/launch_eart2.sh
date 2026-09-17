#!/usr/bin/env bash
# EART v2 (2026-08-21 ~18:50): majority-of-time gripper + crossing interpolation.
# flow30 @ 10 Hz, spline+magscale+eart2, n=50, rp5 & rp10.
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MAGSCALE_K=0.5
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

$PY - <<'EOF' &
import sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_eart2_10env"] = ("flow30_spatial", 10, None, "spline+magscale+eart2")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_eart2_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "eart2_flow10_rp5_out"]
E.main()
EOF
echo "eart2_rp5 pid $!"

$PY - <<'EOF' &
import sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_eart2_10env"] = ("flow30_spatial", 10, None, "spline+magscale+eart2")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_eart2_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--replan_steps", "10", "--out", "eart2_flow10_rp10_out"]
E.main()
EOF
echo "eart2_rp10 pid $!"
echo EART2_LAUNCHED
