#!/bin/bash
# LIBERO headroom screen (SCREEN_RULE.md amendment 1): official lerobot/smolvla_libero, native + zoh k=8,4,2,
# 4 suites x 10 tasks x 5 episodes, dev seeds 5000+.
cd "$(dirname "$0")"; PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python; S=../eval_lerobot_rate.py
run() { t=$1; arm=$2; k=$3; o=libero_${t}_${arm}_k${k}; [ -f $o/eval_info.json ] && return; rm -rf $o  # lerobot refuses an existing output_dir
  MUJOCO_GL=egl RATE_K=$k RATE_ARM=$arm RATE_HOLD=1 $PY $S --policy.path=lerobot/smolvla_libero --env.type=libero --env.task=$t \
    --policy.empty_cameras=1 --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    --eval.n_episodes=5 --eval.batch_size=5 --seed=5000 --policy.device=cuda --output_dir=$o > $o.log 2>&1; }
export -f run; export PY S
T="libero_spatial libero_object libero_goal libero_10"
{ if [ "$1" = conv ]; then for a in spline_satfix tac_fold_satfix qp_anchor; do for t in $T; do echo "$t $a 8"; done; done  # amendment 3
  else for t in $T; do echo "$t native 1"; done; for k in 8 4 2; do for t in $T; do echo "$t zoh $k"; done; done; fi; } | xargs -P ${P:-4} -L1 bash -c 'run $0 $1 $2'
