#!/bin/bash
# Wait for the Spatial@40 asrc run to write its SUMMARY, then run the don_v2 arms.
# Polls the LOG, not pgrep -- the previous chain deadlocked because a stale launcher's
# command line embedded the pgrep pattern, so it waited on a ghost forever.
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1

cat > DONV2_PREREG.md <<'PREREG'
# Pre-registration -- don_v2 @40 Hz (written BEFORE donv2_out/ exists)

Arms, paired n=300 (30 trials x 10 tasks), LIBERO-Spatial, init_state_id pinned + asserted:
  don_v2_native_20env          20 Hz native   -- never run in this campaign
  don_v2_cadmag_spline_40env   40 Hz, cubic spline + cadence + magscale
  don_v2_zoh_40env             40 Hz, zero-order hold

## WHAT THIS CAN AND CANNOT CLAIM -- fixed before any number exists
don_v2 is plain DeepONetHeadV2: forward(prefix, pad_mask). It has NO rate input. Therefore
every off-native don_v2 arm is a harness-side resampler. There is NO don_v2 operator arm, and
a "don_v2 vs spline" cell would be spline compared against itself.
  CANNOT claim: "don_v2 ties/beats cubic spline in multi-rate." Structurally meaningless here.
  CAN claim  : (a) don_v2 RETENTION at 40 Hz vs its own 20 Hz native;
               (b) whether cadence+magscale generalises to a head with zero rate conditioning
                   -- the methodology result, and the strongest version of that claim since
                   don_v2 is architecturally unlike til/asrc;
               (c) ZOH vs cubic spline on this head -- two baselines, a real comparison.

## The head comparison this finally enables
m3_deeponet_s0 and asrc_s0 are BOTH 8300 steps -- budget-matched. So
don_v2_native_20env vs asrc_native_20env (74.7%, 224/300, on disk) is a fair head-vs-head
test at n=300, and it has never been run. Directional expectation is NOT pre-specified;
this is exploratory and will be reported two-sided.

## Decision rules
- Primary comparisons are paired exact McNemar on identical (task, seed) pairs.
- Equivalence statements use TOST via exact Clopper-Pearson CI on the discordant split,
  margin +/- 5.0 pp, same basis as SPATIAL40_PREREG.md (benchmark reruns flip 21.9% of rollouts).
- Stop at n=300. No extension, no task-subset substitution, no retention-ratio framing.
- If don_v2 beats asrc at 20 Hz, that is a statement about HEAD CHOICE, not about rate, and it
  does not license re-running any rate claim on don_v2.
PREREG

# --- wait for the asrc Spatial run to finish ---
for i in $(seq 1 240); do
  grep -q "SUMMARY" spatial40.log 2>/dev/null && break
  sleep 60
done
grep -q "SUMMARY" spatial40.log 2>/dev/null || { echo "[chain] asrc run never finished; aborting" >> chain2.log; exit 1; }

echo "[chain] asrc Spatial@40 done $(date). Starting don_v2 arms." >> chain2.log
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl LIBERO_SUITE=libero_spatial
PY=/home/user/anaconda3/envs/vla_smolvla_libero/bin/python
$PY evaluate_multirate_honest.py \
  --arms don_v2_native_20env,don_v2_cadmag_spline_40env,don_v2_zoh_40env \
  --trials_per_task 30 --suite libero_spatial --out donv2_out \
  > donv2.log 2>&1
echo "[chain] don_v2 done $(date)." >> chain2.log
