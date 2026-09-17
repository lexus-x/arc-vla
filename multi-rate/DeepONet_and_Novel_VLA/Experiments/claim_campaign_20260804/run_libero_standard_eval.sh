#!/usr/bin/env bash
set -euo pipefail

out=${1:?usage: run_libero_standard_eval.sh OUT_DIR NAME=HEAD=CKPT [NAME=HEAD=CKPT ...]}
shift
(($#)) || { echo "at least one NAME=HEAD=CKPT is required" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
export DEEPONET_RATE_HZ=20 TEMPO_SPEED=1.0
export DEEPONET_QUERY_FACTOR=1 DEEPONET_QUERY_POINTS=0
export DEEPONET_FOLD_ALIASES=0 DEEPONET_GAAR=0
export DEEPONET_RETURN_QUERY_GRID=0 DEEPONET_ACTION_INTERP_FACTOR=1

args=()
for model in "$@"; do args+=(--model "$model"); done
exec /home/user/anaconda3/envs/vla_smolvla_libero/bin/python evaluate_libero_standard.py --out "$out" "${args[@]}"
