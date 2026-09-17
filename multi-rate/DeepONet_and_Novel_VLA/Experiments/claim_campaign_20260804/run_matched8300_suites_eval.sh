#!/usr/bin/env bash
# Phase 2b: eval asrc_X vs flow8300_X native 20Hz on object/goal/10, n=50, paired.
set -u
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
BASE=/media/user/C2FE578FFE577A9D/vla_matched

run_one () {
  local suite="$1" s="$2" tag="$3"
  $PY -c "
import sys
sys.path.insert(0, '/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804')
import evaluate_multirate_honest as E
E.MODELS['flow8300_$tag'] = ('flow', '$BASE/flow8300_${s}_s0/checkpoints/8300', None, None)
E.ARMS['flow8300_${tag}_native_20env'] = ('flow8300_$tag', 20, None, None)
sys.argv = ['evaluate_multirate_honest.py',
            '--arms', 'asrc_${tag}_native_20env,flow8300_${tag}_native_20env',
            '--trials_per_task', '5', '--suite', '$suite',
            '--out', 'matched8300_${tag}_out']
E.main()
" || { echo "EVAL FAILED for $suite"; exit 1; }
}

run_one libero_object object object
run_one libero_goal   goal   goal
run_one libero_10     10     long
echo ALL_SUITE_EVAL_DONE
