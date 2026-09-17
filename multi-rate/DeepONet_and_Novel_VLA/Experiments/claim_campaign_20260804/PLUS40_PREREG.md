# Pre-registration — LIBERO-Plus @40 Hz (written BEFORE plus40_out/ exists)

Arms: asrc_cadmag_folding_40env (ours), asrc_anchor_40env (ours/RAI),
asrc_cadmag_spline_40env (comparator). Reference asrc_native_20env = 22.9% (24/105), on disk.

## Hypothesis (directional, fixed in advance)
asrc_s0 consistency_rates = {5,10,25,40,50}. 40 Hz IS in that set, 30 Hz is NOT.
Prediction recorded in evaluate_multirate_honest.py ARMS comment before any of these runs:
ours closes the gap vs spline at 40 Hz and not at 30 Hz. 30 Hz came back a tie (n=300, p=0.23).
One-sided read is therefore legitimate HERE and only here.

## Reporting basis — declared before seeing any 40 Hz number
PRIMARY: all 7 categories, aggregate over 105 tasks. Matches the 30 Hz cell exactly. This is
the number that gets quoted regardless of which basis looks better.
SECONDARY (pre-declared, not post-hoc): 5 categories excluding Robot Initial States and
Sensor Noise. Justification is structural and known in advance, not chosen from the data:
  - Robot Initial States is a VOID cell. The harness pins seed=0 per task so every task gets
    init state 0; the perturbation is never applied. All three 30 Hz arms score 0.0%.
  - Sensor Noise scores 0.0% on all three 30 Hz arms INCLUDING the 20 Hz reference, so it
    carries no signal about rate. Cause undiagnosed.
  These 30 of 105 tasks contribute zero discordant pairs to any paired test.

## Power, stated in advance
Effective n is ~75, not 105. Expect <10 discordant pairs; MDE roughly +/-8 pp one-sided.
This run CAN detect a large effect or bound a small one. It CANNOT certify equivalence.
If |delta| is large and one-sided p<0.05, the follow-up is 3 trials/task (n=315/arm), NOT a
claim on n=105.

## Decision rule
p>=0.05 one-sided -> not significant. Do NOT substitute a category subset, another rate, or
the retention ratio to manufacture a result. The 95.6%/79.0% retention framing at 30 Hz was
4 rollouts; do not repeat it.
