# Pre-registration -- 4-suite 40 Hz sweep (written BEFORE foursuite_*/ exist)

## What changed and why
Every previous cross-suite run used a GLOBAL 11.0 s budget -- the LIBERO-Spatial figure.
Evidence it truncated: in suite_libero_{object,goal,10}.log EVERY rollout ended at the cap
(220/220 @20Hz, 330/330 @30Hz) with ZERO early terminations, while Spatial under the same
220-step budget scores 74.7% with successes finishing at a median of 157 steps. That is a
horizon signature, not a capability signature. The prior vault conclusion "the checkpoint
cannot do these suites" was therefore CONFOUNDED and is being retested, not assumed.

Budget is now per-suite (standard LIBERO max_steps convention at 20 Hz):
  libero_spatial 11.0 s (220) -- UNCHANGED, all existing results preserved bit-for-bit
  libero_object  14.0 s (280)
  libero_goal    15.0 s (300)
  libero_10      26.0 s (520)

## Arms, per suite, n=50 (10 tasks x 5 trials), init_state_id pinned
  asrc_native_20env          -- 20 Hz reference AND the gate
  asrc_cadmag_spline_40env   -- comparator
  asrc_cadmag_folding_40env  -- ours

## GATE, declared before results exist
A suite counts as evaluable ONLY if asrc_native_20env > 0/50 at the corrected horizon.
If the reference is 0/50, that suite's spline-vs-folding cell is VOID -- both arms would be
comparing zero against zero, discordant counts are undefined, and it must NOT be reported as
a tie. Void cells are reported as void.

## What this run can claim
Exploratory, n=50, no directional hypothesis. n=50 CANNOT certify equivalence (~10 discordant
pairs at best). Any suite that clears the gate and shows a gap gets powered to n=300 before a
claim; nothing here is quotable as a result on its own.

## What this run does NOT change
The Spatial @40 Hz result (ours vs spline, -0.3 pp, +/-4.0 pp, n=300, PASS) stands as already
registered in SPATIAL40_PREREG.md and is unaffected -- Spatial's budget is unchanged.
