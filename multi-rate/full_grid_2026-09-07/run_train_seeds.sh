#!/bin/bash
# Train-only DP checkpoints (seeds 0-2, suffix _q2) for the next pre-registered campaign. HEAD=bsp for official B-Spline Policy.
# --n_eval 0: eval slice is empty, no held-out episode is touched. Usage: bash run_train_seeds.sh TASK [TASK...]
cd "$(dirname "$0")"; [ -f env.sh ] && source ./env.sh; PY=${PY:-/home/user/anaconda3/envs/ms3/bin/python}
run() { t=${1% *}; s=${1#* }; h=${HEAD:-step}; L=log_train_${t}_s${s}_q2.txt; [ "$h" != step ] && L=log_train_${t}_${h}_s${s}_q2.txt
  OMP_NUM_THREADS=8 $PY -u harness.py $t --policy dp --steps 30000 --k 4 --n_eval 0 --arms native --seed $s --head $h \
    --checkpoint-suffix _q2 --suffix _trainonly_q2 > $L 2>&1; }
export -f run; export PY HEAD
for t in "$@"; do for s in 0 1 2; do echo "$t $s"; done; done | xargs -P ${P:-4} -I{} bash -c 'run "{}"'
