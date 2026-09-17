# Pre-registration — LIBERO-Spatial @40 Hz, paired n=300 (written BEFORE spatial40_out/ exists)

Arms (all 40 Hz, identical (task,seed) pairs, init_state_id pinned + asserted):
  asrc_cadmag_folding_40env  (ours, operator re-integration)
  asrc_anchor_40env          (ours, RAI = exact rate conversion)
  asrc_cadmag_spline_40env   (comparator, cubic spline)
Reference: asrc_native_20env @20 Hz = 74.7% (224/300), already on disk.

## Why 40 Hz specifically
asrc_s0 consistency_rates = {5,10,25,40,50}. 40 IS in the set, 30 is NOT. The prediction was
recorded in evaluate_multirate_honest.py ARMS before any of these runs. 30 Hz returned a tie
(-3.0 pp, p=0.23, n=300). 40 Hz is the regime where the trained mechanism should apply.

## PRIMARY HYPOTHESIS: EQUIVALENCE, NOT A WIN
The 0.75% offline interpolation bound forbids a meaningful win, so a win is NOT the claim.
Primary test: TOST via exact Clopper-Pearson CI on the discordant split, 90% CI = alpha 0.05.
EQUIVALENCE MARGIN: +/- 5.0 pp, DECLARED NOW, before any 40 Hz Spatial number exists.
Basis for 5.0 pp (pre-existing, not chosen from this data): this benchmark flips 21.9% of
rollouts on an identical rerun (libero-plus-eval-noise-floor). A margin inside the benchmark
own reproducibility is defensible; a margin sized to what n affords is not.

## Stopping rule -- DO NOT OVER-POWER
Stop at n=300 per arm. If the CI does not fit inside +/-5.0 pp, the single permitted extension
is to n=600 and no further. Rationale, stated in advance: the point estimate converges to what
was measured, not to zero. At 30 Hz, projecting RAI to n=1200 gives CI [-3.9, -0.0] -- the tie
would become a significant small deficit purely from added n. Powering past the pre-declared
margin converts an honest equivalence result into a manufactured negative one.

## Secondary (descriptive only, not a claim)
Per-arm success vs the 20 Hz reference (74.7%). Reported, not tested.

## What is NOT permitted
- No substituting a task subset, another rate, or a retention ratio if the primary misses.
- No don_v2 arm in this comparison. don_v2 has no rate input; its only 40 Hz decode is a
  harness-side resampler, so a don_v2-vs-spline 40 Hz cell is spline vs spline. It is
  excluded on structural grounds, decided before the run, not after seeing results.
