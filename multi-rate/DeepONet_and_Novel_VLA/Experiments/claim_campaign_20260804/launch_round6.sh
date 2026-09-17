#!/usr/bin/env bash
# n=300 extension: EART v1 vs spline @10Hz rp5 (resumes from trial 15 in same out dirs)
# + 40Hz magscale k-sweep on asrc-8.3K (folding+cadence, n=50): k in {1.5, 2.0, 2.5}
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
            "--trials_per_task", "30", "--suite", "libero_spatial",
            "--out", "eart150_rp5_out"]
E.main()
EOF
echo "eart300 pid $!"

$PY - <<'EOF' &
import sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.ARMS["flow30_sm_10env"] = ("flow30_spatial", 10, None, "spline+magscale")
sys.argv = ["evaluate_multirate_honest.py", "--arms", "flow30_sm_10env",
            "--trials_per_task", "30", "--suite", "libero_spatial",
            "--out", "spline150_rp5_out"]
E.main()
EOF
echo "spline300 pid $!"

# 40 Hz k-sweep on asrc-8.3K (our module): does k=2.0 under/over-correct?
for K in 1.5 2.5; do
  MAGSCALE_K=$K $PY evaluate_multirate_honest.py \
    --arms asrc_cadmag_folding_40env --trials_per_task 5 --suite libero_spatial \
    --out "k40_$(echo $K | tr '.' 'p')_out" > "k40_$(echo $K | tr '.' 'p').log" 2>&1 &
done
echo "k40 sweep launched"
echo ROUND6_LAUNCHED
