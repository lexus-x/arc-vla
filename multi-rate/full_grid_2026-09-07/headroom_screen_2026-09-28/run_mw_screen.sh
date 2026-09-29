#!/bin/bash
# Meta-World MT10 headroom screen (SCREEN_RULE.md): native + zoh at k=8,4,2, n=50, dev seeds 5000-5049.
cd "$(dirname "$0")"; PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python; S=../eval_lerobot_rate.py
run() { t=$1; arm=$2; k=$3; o=mw_${t}_${arm}_k${k}; [ -f $o/eval_info.json ] && return
  MUJOCO_GL=egl RATE_K=$k RATE_ARM=$arm $PY $S --policy.path=lerobot/smolvla_metaworld --env.type=metaworld --env.task=$t \
    --policy.empty_cameras=2 --rename_map='{"observation.image": "observation.images.camera1"}' \
    --eval.n_episodes=50 --eval.batch_size=10 --seed=5000 --policy.device=cuda --output_dir=$o > $o.log 2>&1; }
export -f run; export PY S
T="reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 window-close-v3"
{ for t in $T; do echo "$t native 1"; done; for k in 8 4 2; do for t in $T; do echo "$t zoh $k"; done; done; } | xargs -P ${P:-8} -L1 bash -c 'run $0 $1 $2'
