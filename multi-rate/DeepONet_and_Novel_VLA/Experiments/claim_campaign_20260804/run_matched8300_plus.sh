#!/usr/bin/env bash
# Matched-budget 8300: asrc_s0 vs flow8300_s0 on LIBERO-Plus, native 20 Hz, paired.
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
$PY - <<'EOF'
import sys
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["flow8300"] = ("flow",
    "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300", None, None)
E.ARMS["flow8300_native_20env"] = ("flow8300", 20, None, None)
import evaluate_plus_multirate as P
sys.argv = ["evaluate_plus_multirate.py",
            "--arms", "asrc_native_20env,flow8300_native_20env",
            "--per_category", "15", "--seed", "0",
            "--out", "plus_matched8300_out"]
P.main()
EOF
