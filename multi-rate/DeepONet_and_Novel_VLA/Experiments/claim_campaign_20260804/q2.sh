#!/bin/bash
# Relaunch the two items that died in q_final.sh.
#   std : died on a MISSING ENV VAR, not a bug -- the provenance guard compared
#         run_config.json's deeponet_head=asrc against an unset DEEPONET_HEAD (default
#         "deeponet"). The guard was right; the launch was wrong. Set all four vars.
#   plus: died on a real bug AFTER the rollouts ran (entry["counters"] held tensors and
#         json.dumps raised). Patched with _jsonable(); rerun from scratch since no
#         results file survived.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=6
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[q2] $(date +%H:%M) $*" >> final2.log; }

log "standard protocol (DEEPONET_HEAD=asrc)"
if [ ! -d std_asrc_out ]; then
  DEEPONET_HEAD=asrc DEEPONET_FOURIER=6 DEEPONET_P=256 DEEPONET_STATE_HISTORY_STEPS=8 \
  $PY evaluate_libero_standard.py \
    --model "asrc=deeponet=$PWD/runs/asrc_s0/checkpoints/8300" \
    --out std_asrc_out > std_asrc.log 2>&1
  log "standard exit=$?"
fi

for A in asrc_cadmag_spline_40env asrc_cadmag_folding_40env; do
  O="pow_plus_${A}_out"; [ -d "$O" ] && continue
  log "plus n=315 $A"
  $PY evaluate_plus_multirate.py --arms "$A" --per_category 45 --out "$O" \
    > "powplus_${A}.log" 2>&1
  log "plus $A exit=$?"
done
log "Q2 DONE"
