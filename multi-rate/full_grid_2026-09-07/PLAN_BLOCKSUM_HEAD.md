# Plan — Block-sum action head + QP decoder ("BSQ") vs B-spline Policy
Written 2026-09-08 14:30 KST, after prior-art + novelty gates (vla-research pipeline stages 1-4).

## 1. FRAME
Question: can Diffusion Policy trained to predict EXACT k-step displacement block sums (lossless coarsening of demos, no
eps fit), decoded by our convex QP (min sum of squared velocity differences s.t. block sums, |v|<=1), match B-spline
Policy at 1X and beat it at 2X/4X speedup on delta-position control, with the same DP backbone and training budget?
Speed dial: one trained model; speedup n = decode each block into 4/n steps (n=1: 4 steps, n=2: 2 steps, n=4: 1 step).
Success bar (from our measured noise: n=100 CI ~ +-10pp, n=400 detects ~7pp): PickCube 4X, BSQ >= B-spline head + 5pp,
paired McNemar per seed, Holm over 3 seeds, seed-mean gap > 2x base seed-stddev. Never significantly below base at 1X.

## 2. PRIOR ART (baseline set) — see prior-art scan
Must beat/match: (a) DP per-step head + naive speedup (ZOH), (b) DP per-step + post-hoc QP (our resampler, exists),
(c) B-spline head (anchor 2607.09648; our reimpl: cubic, 16 control points, uniform clamped knots, LSQ-fit targets).
Reviewer-mandatory later: DiffOG (2504.13807, differentiable QP layer on DP) — stage-2 baseline, not in the first run.
Nearest neighbours that killed alternatives: DiffOG (kills "QP layer in the head"), 2110.04052 + Spline Policy
2606.07386 (kill "velocity-bounded B-spline head"), TempoVLA 2606.06491 (kills "speed-conditioned DP").

## 3-4. IDEATE + NOVELTY (done)
A block-sum head + QP decoder: SURVIVES (no prior exact-block-sum target + convex feasibility decoder + k as speed dial).
B feasibility-projected time-compressed training: WOUNDED (crowded: SAIL, TempoVLA, RACE, SpeedAug) — keep as add-on ablation only.
C/D/E: KILLED. Residual claim for A rests on the empirical 4X win; the mechanism is the already-validated satfix/QP effect,
now trained in. If the trained-in part adds nothing over post-hoc QP (ablation 7b), the representation claim dies.

## 5. IMPLEMENT (minimal, dp_min.py + harness.py)
- make_chunks target option: coarsen 16-step action window into B=4 block sums (k_train=4); gripper/hold dims: last value per block.
- DP horizon = B (U-Net over 4 blocks); everything else unchanged. Flag: --head {step,blocksum,bspline}.
- Decode: resample_qp(block_sums, k=4/n); n=4 -> k=1 -> clip. Replan every 2 blocks (= 8 native steps at 1X, as now).
- B-spline head: predict 16 control points; targets = fixed-knot LSQ fit of cumulative positions; decode by basis eval at 16/n points.
~150 lines total. Self-check: blocksum head at k=4 on a demo chunk reconstructs demo displacements exactly.

## 6. BASELINE (reproduce first)
Train base DP 3 seeds on PushT, PickCube -> seed stddev at 1X (defines "improvement"). Train B-spline head 3 seeds; must land
within 5pp of base at 1X (paper: PushT +3, Lift 0). If it does not, fix the reimplementation before any BSQ comparison.

## 7. ABLATE (one variable at a time, 3 seeds, n=400 on PushT/PickCube; Lift/Can/Square 1 seed n=100 at 1X only)
7a head: step vs blocksum vs bspline, all at speeds 1X/2X/4X.        (main table)
7b decoder for blocksum head: ZOH vs satfix vs QP at 4X.              (is the decoder the mechanism?)
7c trained-in vs post-hoc: blocksum head+QP vs step head+post-hoc QP. (does training on block sums add anything?)
7d B in {2,4,8} blocks per horizon, 1 seed.                            (resolution sensitivity)
7e (optional, candidate B) blocksum head trained on satfix-projected 4X targets.

## 8. VERIFY — pre-registered predictions (sealed before any training)
P1 primary: PickCube 4X: BSQ > B-spline head by >= 5pp (Holm over 3 seeds, m=3).
P2: PushT 2X: BSQ >= B-spline head (direction); PushT 4X void if all arms < 10%.
P3 no-harm: 1X, all 5 tasks: BSQ within 3pp of base and of B-spline head; never significantly below.
P4 mechanism: 7b QP > ZOH at 4X; 7c BSQ > step+post-hoc QP at 4X by >= 3pp.
Verdicts: P1 fails -> KILL. P1 holds, P4(7c) fails -> REVISE: claim shrinks to "post-hoc QP suffices, no retraining"
(publishable as a negative result against B-spline Policy's retraining). P1+P3+P4 hold -> PUBLISH candidate.
Not allowed after results: changing B, k_train, n, eps of the B-spline baseline, dropping seeds or cells.

## Compute
Training ~10 min/task/seed here. 2 tasks x 3 heads x 3 seeds = 18 runs (~3 h, 2 streams -> 1.5 h) + 7d/7e ~8 runs.
Eval n=400: PushT ~28 min/arm, PickCube ~10 min/arm; 27 arms/task main + ~12 ablation arms -> ~20 h serial, ~5 h on 4 streams.
Ceiling tasks 1X n=100: ~1 h. Total: 1 day implementation + baseline reproduction, 1 day ablations/eval. RoboCasa: not in scope.
Deviation 2026-09-08 16:20 (before any PushT head result): PushT 4X runs skipped for seeds 1,2 under the pre-reg void rule (earlier campaign: all arms 1-3% at 4X vs native 26%); seed 0 stream keeps them.

## Stage 6/7 findings, 2026-09-08 20:20 (analyze_c2.py, n=100 x 3 seeds)
- Stage 6 PASSED: B-spline head at 1X = 90 +- 1 on PickCube vs base 90 +- 0 (seed sd ~0 at 1X).
- SEMANTIC MISMATCH (implementation error in the plan): step-head "kX" rows are MULTI-RATE (policy at 1/k rate, block sums
  resampled back to k fine steps, same wall time). heads.decode implements EXECUTE-FASTER (block executed in kb/n env steps,
  B-spline sampled at 16/n points). The two "2X/4X" columns are different experiments and must not be compared.
- No headroom: PickCube demos already saturate the clip; 2-step sums exceed |1| in 40% of elements (29% displacement lost),
  4-step sums 61% (54% lost); PushT 31%/44%. Execute-faster is physically impossible on these RL demos: B-spline head 2%/1%,
  block-sum head 2-3% at 2X. This is a task property, not a method property; B-spline Policy's speedup needs teleop demos with headroom.
- Block-sum head at "1X" = 52 +- 2 = step head + post-hoc QP at k=4 (51 +- 1): the head is a k=4-coarse policy by construction,
  cannot be a 1X method, and the trained-in coarse target adds nothing over post-hoc coarsening (P4b answered: no gain).
VERDICT candidate A: KILL as a representation claim. What stands: the post-hoc QP/satfix multi-rate result (51 vs 41 ZOH at k=4, 3 seeds).
Next (needs decision): execute-faster comparison on tasks WITH headroom (RoboMimic lift/can/square, teleop demos, |a|>1 = 0 at 1X):
B-spline head vs base+clip vs base+qp_eps path-following decoder at 2X/4X. Caches being built (log_dp_*_k1_cache.txt).
Deviation 21:20: PushT blocksum runs and seed-0 step 4X cancelled (candidate A killed; 4X void); only B-spline head 1X kept.
