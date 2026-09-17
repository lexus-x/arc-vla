# Pre-registration -- per-suite 40 Hz evals (written BEFORE suiteeval_*/ exist)

Checkpoints: asrc_{object,goal,long}_s0 @8300, each trained ONLY on its own suite, config
replicated from asrc_s0 (only --dataset/--out differ). Budget-matched at 8300 steps.

Arms per suite, n=50 (10 tasks x 5 trials), init_state_id pinned, per-suite horizon
(object 280 / goal 300 / long 520 steps at 20 Hz; doubled at 40 Hz):
  <m>_native_20env          20 Hz reference AND GATE
  <m>_cadmag_spline_40env   comparator
  <m>_cadmag_folding_40env  ours

Normalization stats follow the CHECKPOINT via MODEL_DATASET (asrc_object ->
libero_object_image etc), not the eval suite. This matters because
rate_integrated_deeponet.py:130 uses action_scale INSIDE the RK4 integration, so wrong stats
change the head's dynamics rather than merely rescaling output. Each arm prints a [stats] line.

## GATE, declared before any result exists
A suite counts ONLY if <m>_native_20env > 0/50 on its own suite. A suite-trained checkpoint
reading 0/50 on the suite it was trained on is a TRAINING FAILURE, not a rate finding, and must
be reported as such. A 0-vs-0 spline-vs-folding cell is VOID, never a tie.

## What this can claim
n=50 is a GATE, not a result. ~10 discordant pairs at best cannot certify equivalence, and this
project has twice seen n=50 estimates reverse under power (spline@40 78%->72%, RAI@40 66%->71%).
Cells that clear the gate get powered to n=300 before anything is claimed. No directional
hypothesis is pre-specified for the new suites; report two-sided.

## Known caveat, stated in advance
Budget-matching at 8300 steps gives each suite a different number of data passes:
object 5.95 epochs, goal 7.66, long ~3.9 (101k frames). A weak `long` result is therefore
ambiguous between "no decoding benefit" and "undertrained", and cannot be attributed without a
second long run at higher budget.
