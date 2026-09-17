# Pre-registration: powered ours-vs-spline test (2026-08-11)

## Why now
Three independent estimates of the ours-vs-spline gap agree: +4.0 pp (folding@30, n=150),
+4.0 pp (RAI@30, n=50), +3.8 pp (LIBERO-Plus, n=105). Noise scatters around zero; it does not
land on +4 pp three times. n=150 gave b:c = 10:4, which cannot reach significance regardless of
how real the effect is. This test is powered to resolve it.

## Design (fixed BEFORE the run)
- Arms: asrc_cadmag_folding_30env (ours), asrc_cadmag_spline_30env (comparator),
  asrc_native_20env (20 Hz reference, reported alongside per the standing rule).
- 30 trials/task x 10 libero_spatial tasks = n=300 per arm. Fresh out dir, no reuse.
- Matched wall-clock 11.0 s, replan 0.5 s, init_state_id pinned to trial index with an assert.
- Primary test: exact McNemar on ours vs spline over identical (task, seed) pairs.
- Power: at the observed 10:4 rate, n=300 -> b:c ~ 20:8 -> p ~ 0.038. n=450 would give p ~ 0.009.

## Decision rule (fixed BEFORE the run)
- p < 0.05 -> the effect is real; report it with the effect size and the reference alongside.
- p >= 0.05 -> report as NOT significant and stop powering this comparison. Do not extend n
  further looking for a crossing, and do not report a subset of tasks or the 40 Hz cell instead.
- Report the number that comes out either way. No cherry-picking of cells, rates, or categories.

## Known ceiling
Spline is within 0.75% of EXACT operator rate conversion (measured, 64 contexts), and RAI --
which removes the approximation entirely -- scored +4.0 pp / p=0.73. So even a confirmed +4 pp
is a small effect near a hard bound, not a decoding breakthrough. It is reported as such.

---

# Pre-registration 2: powered til-vs-asrc base effect (2026-08-11)

## Result of test 1
ours 71.7% vs spline 74.7% vs 20 Hz reference 74.7% (n=300 each). Paired: ours-vs-spline
-3.0 pp p=0.2327; ours-vs-reference -3.0 pp p=0.2717; spline-vs-reference +0.0 pp p=1.0000
(25 discordant each way). Per the fixed rule, p>=0.05 -> NOT significant, stop powering the
ours-vs-spline comparison. Decoding route closed.

## Why test 2
The only large effect in the campaign is asrc over til (+16 to +26 pp at every rate). But
til_native_20env has only n=50 (60.0%, CI ~ +-14 pp), so the effect size is not established.
Attributing it with a 2 h ablation training run before measuring it would be backwards.

## Design (fixed BEFORE the run)
- Arm: til_native_20env, 30 trials/task x 10 libero_spatial tasks = n=300, into powered_out.
- Paired against the already-complete asrc_native_20env (n=300, same tasks, same seeds,
  init_state_id pinned to trial index) via exact McNemar on identical (task, seed) pairs.
- Matched wall-clock 11.0 s, replan 0.5 s. No transform on either arm, so this isolates the
  HEAD/TRAINING difference with no rate content at all.

## Decision rule (fixed BEFORE the run)
- If the paired gap is significant, the effect is real and the 3-way confound (consistency loss
  vs n_fourier 6-vs-0 vs trunk_bandlimit) is worth a 2 h ablation to attribute. Relaxing the
  train.py:339 guard is then justified.
- If it is NOT significant, the campaign has no large effect either, and no ablation is warranted.
- Report the number either way. This is a base-performance comparison at 20 Hz with NO multi-rate
  content; it must not be presented as a multi-rate result.

---

# Pre-registration 3: powered RAI (rate-anchored) arm (2026-08-11)

## Results of tests 1-2
Test 1: ours(folding) 71.7% vs spline 74.7% vs 20 Hz ref 74.7% (n=300 each). Paired
folding-vs-spline -3.0 pp p=0.2327; spline-vs-ref +0.0 pp p=1.0000. Decoding win: NO.
Test 2: asrc 74.7% vs til 68.3% (n=300), +6.3 pp, 54 vs 35, p=0.0558. Not significant ->
ablation NOT warranted (attributing 6.3 pp across 3 factors needs n~1000+/arm).

## Why test 3
folding integrates with the trunk rate token at log2(30/20)=0.585, which asrc never trained on
(consistency_rates = 5,10,25,40,50). RAI pins the token to 20 while integrating at dt=1/30, so the
field is queried in distribution. Independent evidence the token matters: RAI at 40 Hz scores 66.0%
vs folding 74.0%, because 40 Hz IS trained and pinning to 20 discards it. RAI at 30 Hz has only
n=50 (74.0%).

This is NOT an extension of a negative result -- it is a different arm with a mechanism. Test 1's
stop powering rule applied to the folding-vs-spline comparison, which stays closed.

## Hypothesis (fixed BEFORE the run) -- EQUIVALENCE, not a win
The 0.75% interpolation bound means RAI CANNOT beat spline by a meaningful margin. The question is
whether RAI removes folding's 3.0 pp deficit, i.e. whether operator decoding is EQUIVALENT to cubic
spline at an off-native rate once rate-anchored. A tie is the predicted and acceptable outcome.

## Design
- Arm: asrc_anchor_30env, 30 trials/task x 10 libero_spatial tasks = n=300, into powered_out.
- Paired via exact McNemar against the completed asrc_cadmag_spline_30env (n=300) AND against
  asrc_native_20env (n=300) on identical (task, seed) pairs. init_state_id pinned.
- Verify anchor_applied > 0, magscale_applied > 0, cadence queries = 2/3 of env steps before
  trusting the arm.

## Decision rule (fixed BEFORE the run)
- RAI vs spline p >= 0.05 AND |delta| small -> report as EQUIVALENT. This is the campaign's final
  statement on decoding. Do not power any further decoding arm.
- RAI significantly WORSE than spline -> operator decoding is inferior, report that.
- RAI significantly BETTER -> unexpected given the 1% bound; re-verify counters and the bound
  before believing it.
- Report the number either way. No cherry-picking rates or tasks.

---

# Pre-registration 4: matched-budget HEAD comparison (2026-08-11)

## Why the comparison has to change
Tests 1-3 closed the decoder question: spline sits 0.75% from EXACT operator conversion, RAI
(exact) tied it, and every operator arm lands at or below spline (spline 74.7% == native-20Hz
74.7%, p=1.0000). ours and spline read the SAME chunk from the SAME policy, so they are bounded
~1% apart by construction. No decoder can win. The only available win is a better HEAD, evaluated
against flow + cubic spline.

This is a DIFFERENT CLAIM and must be labelled as such. It is NOT "our decoding beats cubic
spline"; it is "our head beats the flow + cubic-spline pipeline at off-native rates". The spline
is doing its job perfectly in both readings.

## Budget confound, and why these checkpoints
FLOW_CKPT is flow_s0 @30000 steps vs asrc @8300 -- a 3.6x budget gap FAVOURING the baseline
(budget is worth up to +26 pp on LIBERO-Plus). Using it would be an unfair comparison in our
favour if we won, and uninformative if we lost. m1_flow_s* and m3_deeponet_s* are ALL 8300, so
they are the only fair pair available. Prior verified result on this pair: don_v2 +20.9 pp over
flow at matched 8.3K, 5 seeds/arm, p<1e-5 -- but that was measured in a DIFFERENT harness, so it
must be reproduced here before anything is built on it.

## Design (fixed BEFORE the run)
- Arms: don_v2_native_20env (m3_deeponet_s0 @8300) and flow_m1_native_20env (m1_flow_s0 @8300).
- 15 trials/task x 10 libero_spatial tasks = n=150 per arm. A +20 pp effect is overwhelming at
  this n; if it needs more than n=150 to see, it is not the effect the prior work claimed.
- Matched wall-clock 11.0 s, replan 0.5 s, init_state_id pinned, native 20 Hz for both (no rate
  content at all in this test).
- Paired exact McNemar on identical (task, seed) pairs.
- The load path has a hard shape guard ("model head and checkpoint tensors differ; refusing
  random init"), which is what catches the deeponet_head_v2 import-shadowing hazard -- the two
  copies of that file on disk have different md5s.

## Decision rule (fixed BEFORE the run)
- don_v2 beats flow by a large, significant margin -> the head advantage is real in THIS harness,
  and building 30 Hz arms on top is justified. Report as a head result, never as a decoder result.
- Margin small or not significant -> the prior +20.9 pp does not reproduce under the corrected
  protocol, and there is no win available anywhere. Say so and stop.
- Report the number either way. Also report task 5 specifically, since it is 1/15 for every
  asrc/til arm and is where the grasp-height mechanism should show up.
