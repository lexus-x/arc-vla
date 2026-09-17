#!/bin/bash
# Four remaining open items, sequential. Training-grade VRAM is not involved;
# these are evals, but run one at a time so the 1000+ step cells never contend.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=6
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
log(){ echo "[final] $(date +%H:%M) $*" >> final.log; }

for i in $(seq 1 480); do pgrep -f "[t]rain.py" >/dev/null || break; sleep 60; done

# 1. GOAL powered to n=300  (the cell with the largest nominal lead, +6.0 pp)
for A in asrc_goal_cadmag_spline_40env asrc_goal_cadmag_folding_40env; do
  O="pow_goal_${A}_out"; [ -d "$O" ] && continue
  log "goal n=300 $A"
  LIBERO_SUITE=libero_goal $PY evaluate_multirate_honest.py --arms "$A" \
    --trials_per_task 30 --suite libero_goal --out "$O" > "pow_${A}.log" 2>&1
done

# 2. ABLATION of the two harness fixes, 2x2, n=50
log "ablation 2x2"
for A in asrc_folding_40env asrc_cadence_folding_40env asrc_mag_folding_40env asrc_cadmag_folding_40env; do
  O="abl_${A}_out"; [ -d "$O" ] && continue
  LIBERO_SUITE=libero_spatial $PY evaluate_multirate_honest.py --arms "$A" \
    --trials_per_task 5 --suite libero_spatial --out "$O" > "abl_${A}.log" 2>&1
done

# 3. STANDARD PROTOCOL number for asrc_s0 (locked: 220 steps, n_action_steps=1)
log "standard protocol"
[ -d std_asrc_out ] || $PY evaluate_libero_standard.py \
  --model "asrc=deeponet=$PWD/runs/asrc_s0/checkpoints/8300" \
  --out std_asrc_out > std_asrc.log 2>&1

# 4. PLUS powered: 45 tasks/category x 7 = 315
for A in asrc_cadmag_spline_40env asrc_cadmag_folding_40env; do
  O="pow_plus_${A}_out"; [ -d "$O" ] && continue
  log "plus n=315 $A"
  $PY evaluate_plus_multirate.py --arms "$A" --per_category 45 --out "$O" \
    > "powplus_${A}.log" 2>&1
done
log "ALL DONE"
