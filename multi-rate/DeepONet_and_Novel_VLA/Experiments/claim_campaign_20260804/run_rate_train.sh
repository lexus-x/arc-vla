#!/usr/bin/env bash
set -euo pipefail

head=${1:?usage: run_rate_train.sh ti|til|asrc|tempo|ncde_style OUT_DIR}
out=${2:?usage: run_rate_train.sh ti|til|asrc|tempo|ncde_style OUT_DIR}
case "$head" in
  ti) extra=(--rate_consistency_weight 0) ;;
  til) extra=(--rate_consistency_weight 0) ;;
  asrc) extra=(--trunk_bandlimit --rate_consistency_weight "${RATE_WEIGHT:-0.1}"
               --consistency_rates "${CONSISTENCY_RATES:-5,10,25,40,50}") ;;
  tempo) extra=(--tempo_min_speed "${TEMPO_MIN_SPEED:-0.5}"
                 --tempo_max_speed "${TEMPO_MAX_SPEED:-2.0}") ;;
  ncde_style) extra=(--rate_consistency_weight 0) ;;
  *) echo "head must be ti, til, asrc, tempo, or ncde_style" >&2; exit 2 ;;
esac

stage1=${SCREEN_STEPS:-500}
stage2=${SCREEN_STAGE2_STEPS:-0}
ckpt_every=${SCREEN_CKPT_EVERY:-$((stage1 + stage2))}
base_ckpt=${BASE_CKPT:-lerobot/smolvla_base}

exec /home/user/anaconda3/envs/vla_smolvla_libero/bin/python train.py \
  --head deeponet --variant baseline --deeponet_head "$head" \
  --base_ckpt "$base_ckpt" \
  --deeponet_p "${SCREEN_P:-256}" --deeponet_blocks "${SCREEN_BLOCKS:-1}" \
  --deeponet_queries "${SCREEN_QUERIES:-8}" --deeponet_fourier "${SCREEN_FOURIER:-6}" \
  --stage1_steps "$stage1" --stage2_steps "$stage2" \
  --stage1_batch "${SCREEN_BATCH:-16}" --stage2_batch "${SCREEN_BATCH:-16}" \
  --num_workers "${SCREEN_WORKERS:-4}" --ckpt_every "$ckpt_every" \
  --out "$out" "${extra[@]}"
