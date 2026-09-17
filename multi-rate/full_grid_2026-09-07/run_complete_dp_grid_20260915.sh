#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
MS=/home/user/anaconda3/envs/ms3/bin/python
RM=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
common=(--policy dp --seed 0 --steps 30000 --n_train 200 --n_eval 100 --workers 4)

run_step() {
  local py=$1 task=$2 k=$3 arms=$4
  "$py" -u harness.py "$task" "${common[@]}" --head step --k "$k" --arms "$arms" --suffix _matrix_missing
}

run_bspline_head() {
  local py=$1 task=$2 k=$3
  "$py" -u harness.py "$task" "${common[@]}" --head bspline --k "$k" --arms native --suffix _matrix
}

# PushT: k=2 is already complete.
run_step "$MS" PushT-v1 1 spline,tac_fold_satfix,qp,qp_anchor
run_step "$MS" PushT-v1 4 spline,bspline_eps_raw,qp_anchor
run_step "$MS" PushT-v1 8 native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor

# RoboMimic: fill only absent arms from standard n=100 results.
run_step "$RM" lift 1 spline,tac_fold_satfix,qp,qp_anchor
run_step "$RM" lift 2 bspline_eps_raw,qp,qp_anchor
run_step "$RM" lift 4 spline,bspline_eps_raw,qp_anchor
run_step "$RM" lift 8 native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor

for task in can square; do
  run_step "$RM" "$task" 1 spline,tac_fold_satfix,qp,qp_anchor
  run_step "$RM" "$task" 2 bspline_eps_raw,qp_anchor
  run_step "$RM" "$task" 4 spline,bspline_eps_raw,qp_anchor
  run_step "$RM" "$task" 8 native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor
done

# Trained B-spline Policy: separate execute-faster table.
for k in 2 4 8; do
  run_bspline_head "$MS" PushT-v1 "$k"
done
for task in lift can square; do
  for k in 2 4 8; do
    run_bspline_head "$RM" "$task" "$k"
  done
done
