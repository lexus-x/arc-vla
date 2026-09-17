#!/usr/bin/env bash
# Round 5 (2026-08-21 ~20:10): ft40 @ 40 Hz folding eval (n=50) once the ckpt lands.
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

for i in $(seq 1 60); do
  [ -d runs/asrc40hz_ft_s0/checkpoints/2000 ] && break
  sleep 30
done

FT40="/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/runs/asrc40hz_ft_s0/checkpoints/2000"

$PY - "$FT40" <<'EOF'
import sys
ft = sys.argv[1]
sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
E.MODELS["asrc"] = ("deeponet", ft, "asrc", 6)
sys.argv = ["evaluate_multirate_honest.py", "--arms", "asrc_cadmag_folding_40env",
            "--trials_per_task", "5", "--suite", "libero_spatial",
            "--out", "ft40_fold40_out"]
E.main()
EOF
echo ROUND5_DONE
