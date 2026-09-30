# PREREG — ARC vs best-in-field (Spline, B-Spline), 4-bench campaign

Written 2026-09-29 BEFORE launching. ARC = `qp_anchor`. "Best in field" = cubic spline family's
best arm and B-Spline Policy Alg.1 (`bspline_eps`), each family's best per bench.

## Primary claim (the publication bar)
ARC achieves the best average success rate AND a positive (beats both spline families)
sim-bench verdict on >= 4 sim suites. A suite is positive when the suite-mean favors ARC over
both families' best arms, with >= 1 Holm-significant task-level win and no Holm-significant loss
inside the suite.

## Void / screening rules (fixed before data)
- A bench is evaluable only if the policy's raw action saturation frac > 0.05 (mechanism engages)
  and native success is in [30%, 90%] (room to move) at the tested k. Benches failing this screen
  are reported as VOID, never as losses or wins (RoboMimic/RoboCasa at sat=0.00 are expected void).
- Holm correction across the full task x k family of the campaign. Exact two-sided McNemar.
- k=4 primary; k=2 secondary; PushT k=3 pre-declared as the spline-overshoot rate point.

## Phase 1 (now): complete ManiSkill pairing (bench #1)
6 tasks, same windows/checkpoints as the sealed confirm runs (n_train=200, offset=100, n=400,
seed=0, k=4), arms: native,zoh,spline,spline_satfix,bspline_eps_raw,qp_anchor, suffix _splinefaceoff:
RollBall-v1, LiftPegUpright-v1, PushCube-v1, AnymalC-Reach-v1, PokeCube-v1, StackCube-v1.
(PickCube offset=500 n=293 and PullCube already paired.) Pairing validity: qp_anchor vectors must
match the sealed confirm files exactly (same checkpoint/seed/window); mismatch voids the pairing.

## Phase 2: screen new benches (benches #3-4 candidates)
RoboTwin / VLABench / CALVIN / LIBERO / MetaWorld: train/load a DP-style policy, measure
sat_frac + native success, keep only benches passing the screen, then run the paired face-off at
k in {2,3,4,8} and declare per-screen results.

## Phase 3: power upgrade on PushT (bench #2)
PushT k=3 n=400 -> n=800 if margin needs resolution. k=2 loss must be reported regardless.

## Not allowed after seeing results
No arm changes, no k changes, no dropping benches, no re-defining "positive". A failed screen is
reported as VOID; a loss is reported as a loss.
