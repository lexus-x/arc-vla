#!/bin/bash
# PREREG_LEARNED_GOVERNOR.md confirmation, k=4. Usage: bash run_confirm.sh TASK [TASK...]  (max $P concurrent, default 3)
cd "$(dirname "$0")"; PY=${PY:-/home/user/anaconda3/envs/ms3/bin/python}
ARMS=native,zoh,tac_fold_satfix,qp_anchor,qp_learned,learned_raw,learned_tanh,mlp_bc
run() { t=$1; if [ "$t" = PickCube-v1 ]; then w="--eval-offset 500 --n_eval 293"; else w="--eval-offset 100 --n_eval 400"; fi
  $PY -u harness.py $t --policy dp --steps 30000 --k 4 $w --workers 4 --arms $ARMS --suffix _confirm > log_confirm_${t}_k4.txt 2>&1; }
export -f run; export PY ARMS
printf '%s\n' "$@" | xargs -P ${P:-3} -I{} bash -c 'run {}'
