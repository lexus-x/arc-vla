#!/bin/bash
# RoboTwin 2.0 headroom screen (SCREEN_RULE.md amendment 2): official lerobot/smolvla_robotwin, 10 lerobot-CI tasks,
# n=50, dev seeds 5000-5049. Staged: native on all tasks -> zoh on tasks with native >= 30% -> k=2,4,8, stop at first pass.
cd "$(dirname "$0")"; H=$PWD; R=/home/user/Desktop/multi-rate/external/RoboTwin; OUT=$H/run_robotwin_screen.out
PY=/home/user/anaconda3/envs/robotwin/bin/python; S=$H/../eval_lerobot_rate.py
RM='{"observation.images.head_camera": "observation.images.camera1", "observation.images.left_camera": "observation.images.camera2", "observation.images.right_camera": "observation.images.camera3"}'
run() { t=$1; arm=$2; k=$3; o=$H/rt_${t}_${arm}_k${k}; [ -f $o/eval_info.json ] && return; rm -rf $o  # lerobot refuses an existing output_dir
  L=$(awk -F': *' -v t=$t '$1==t{print $2}' $R/task_config/_eval_step_limit.yml)
  cd $R && PYTHONPATH=$R RATE_K=$k RATE_ARM=$arm RATE_ABS=1 RATE_HOLD_IDX=6,13 $PY $S --policy.path=lerobot/smolvla_robotwin \
    --env.type=robotwin --env.task=$t --env.episode_length=$L --eval.n_episodes=50 --eval.batch_size=1 --eval.use_async_envs=false \
    --rename_map="$RM" --seed=5000 --policy.device=cuda --output_dir=$o > $o.log 2>&1; }
export -f run; export H R PY S RM
stage() { for i in 1 2; do echo "$1" | xargs -P ${P:-10} -L1 bash -c 'run $0 $1 $2'; done; }  # pass 2 reruns crashed cells
pc() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['overall']['pc_success'])" $H/rt_$1/eval_info.json 2>/dev/null || echo -1; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>=b)}'; }

T="beat_block_hammer click_bell handover_block stack_blocks_two click_alarmclock open_microwave adjust_bottle lift_pot stamp_seal turn_switch"
echo "start $(date)" >> $OUT
stage "$(for t in $T; do echo "$t native 1"; done)"
E=""; for t in $T; do n=$(pc ${t}_native_k1); echo "native $t $n" >> $OUT; ge $n 30 && E="$E $t"; done
nE=$(echo $E | wc -w); echo "native >= 30%: $nE/10 ($E)" >> $OUT
[ $nE -lt 5 ] && { echo "VERDICT: fail (only $nE tasks with native >= 30%) $(date)" >> $OUT; exit; }
for k in 2 4 8; do
  stage "$(for t in $E; do echo "$t zoh $k"; done)"
  w=0; for t in $E; do n=$(pc ${t}_native_k1); z=$(pc ${t}_zoh_k$k); echo "zoh k=$k $t $n $z" >> $OUT
    ge $z 0 && ge $(awk -v a=$n -v b=$z 'BEGIN{print a-b}') 10 && w=$((w+1)); done  # z=-1 (missing) never counts
  echo "k=$k: $w/10 tasks with native - zoh >= 10 pp" >> $OUT
  [ $w -ge 5 ] && { echo "VERDICT: pass at k=$k $(date)" >> $OUT; exit; }
done
echo "VERDICT: fail (no k passes) $(date)" >> $OUT
