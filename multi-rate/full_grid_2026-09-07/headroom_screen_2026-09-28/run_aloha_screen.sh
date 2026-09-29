#!/bin/bash
# ALOHA transfer-cube headroom screen (SCREEN_RULE.md): native + zoh at k=8,4,2, n=50, dev seeds 5000-5049.
# Official lerobot/act_aloha_sim_transfer_cube_human migrated to processor format (../hub_act_aloha_transfer_cube).
# Absolute joint targets: displacement path from current qpos; grippers (dims 6, 13) causal-held.
cd "$(dirname "$0")"; PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python; S=../eval_lerobot_rate.py
run() { arm=$1; k=$2; o=aloha_transfer_cube_${arm}_k${k}; [ -f $o/eval_info.json ] && return
  MUJOCO_GL=egl RATE_K=$k RATE_ARM=$arm RATE_ABS=1 RATE_HOLD_IDX=6,13 $PY $S --policy.path=../hub_act_aloha_transfer_cube \
    --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=50 --eval.batch_size=10 --seed=5000 \
    --policy.device=cuda --output_dir=$o > $o.log 2>&1; }
export -f run; export PY S
printf "native 1\nzoh 8\nzoh 4\nzoh 2\n" | xargs -P ${P:-2} -L1 bash -c 'run $0 $1'
