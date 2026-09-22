#!/bin/bash
# Run 50 Hz Push-T eval for arc, then qp_anchor + tac_fold_satfix
set -e
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
PYTHON=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python

echo "=== Starting ARC 50Hz eval ==="
$PYTHON run_pusht_rate.py --rate 50.0Hz --arms arc --n_episodes 50 --batch_size 10 --start_seed 1000 --out_file eval_pusht_arc_50hz.json
echo "=== ARC 50Hz complete ==="

echo "=== Starting QP+TAC 50Hz eval ==="
$PYTHON run_pusht_rate.py --rate 50.0Hz --arms qp_anchor,tac_fold_satfix --n_episodes 50 --batch_size 10 --start_seed 1000 --out_file eval_pusht_qp_50hz.json
echo "=== QP+TAC 50Hz complete ==="

echo "=== ALL 50Hz EVALS DONE ==="
