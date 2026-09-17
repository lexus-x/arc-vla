#!/usr/bin/env bash
# Round 4b (2026-08-21 ~19:00): FAIR flow@10Hz baseline = spline + magscale k=0.5
# (the spline-only cell over-travels: its x2 displacement rescale stacks with the
#  OSC plant's larger per-command gain at 10 Hz — same physics magscale fixes for us).
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MAGSCALE_K=0.5
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

$PY - <<'EOF' &
import os, sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_sm_10env"] = ("flow30_spatial", 10, None, "spline+magscale")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_sm_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "flow10sm_rp5_out"]
E.main()
EOF
echo "flow10sm_rp5 pid $!"

$PY - <<'EOF' &
import os, sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_sm_10env"] = ("flow30_spatial", 10, None, "spline+magscale")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_sm_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--replan_steps", "10", "--out", "flow10sm_rp10_out"]
E.main()
EOF
echo "flow10sm_rp10 pid $!"
echo ROUND4B_LAUNCHED
