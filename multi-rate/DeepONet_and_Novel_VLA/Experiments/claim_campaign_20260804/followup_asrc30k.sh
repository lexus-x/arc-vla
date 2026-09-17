#!/usr/bin/env bash
# Auto-followup (2026-08-21 late): when asrc30k_s0 training finishes, evaluate
# native 220/10 (n=100), 10Hz k=0.5 rp5+rp10 (n=50), 40Hz folding (n=50).
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

CKPT=runs/asrc30k_s0/checkpoints/30000
for i in $(seq 1 960); do
  [ -d "$CKPT" ] && break
  sleep 60
done
[ -d "$CKPT" ] || { echo "TIMED OUT waiting for $CKPT"; exit 1; }

FT="$PWD/$CKPT"

$PY - "$FT" <<'EOF'
import sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_native_20env",
            "--trials_per_task", "10", "--suite", "libero_spatial",
            "--out", "asrc30k_native_out"]
E.main()
EOF

$PY - "$FT" <<'EOF'
import os, sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
os.environ["MAGSCALE_K"] = "0.5"
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_mag_folding_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "asrc30k_10hz_rp5_out"]
E.main()
EOF

$PY - "$FT" <<'EOF'
import os, sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
os.environ["MAGSCALE_K"] = "0.5"
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_mag_folding_10env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--replan_steps", "10", "--out", "asrc30k_10hz_rp10_out"]
E.main()
EOF

$PY - "$FT" <<'EOF'
import sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_cadmag_folding_40env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "asrc30k_40hz_out"]
E.main()
EOF
echo FOLLOWUP_ALL_DONE
