# Pre-registration — global constrained-QP resampler vs satfix family, PickCube open-loop replay
Written 2026-09-07 22:56 KST, BEFORE any success-rate result for the QP arm exists.

## Method under test (ours)
`resample_qp` (`resample_qp.py`): per action dimension, solve one convex QP over the whole episode:
minimize sum of squared consecutive differences of the fine-rate velocity, subject to (a) every
k-block's sum equals the true block sum exactly, (b) every fine-rate value in [-1,1]. Feasible by
construction; no post-hoc satfix. Differs from every existing arm in that the controller clip is a
hard constraint of the fit, coupled across blocks, rather than ignored (spline family) or patched
one block at a time afterward (satfix = the per-block L2 projection; we proved the global L2
projection decomposes to exactly satfix, so any gain must come from a cross-block objective).

## Evidence motivating this test (no success rates seen)
`ksweep_results.json` (60 real PickCube demos, reconstruction MSE, no sim): QP vs cubic_spline_satfix
−4.7% (k=2), −7.7% (k=3), −16.4% (k=4), −15.2% (k=5); QP is the best arm of all at k≥3. Raw spline
violation rate 39–51% of steps. The vault's own diagnostic (2026-09-06) found MSE ranking predicts
success ranking on this data family.

## Protocol (identical to satfix_2026-09-05/run_satfix_general.py, which produced the k=2/k=4 baselines)
- ManiSkill PickCube-v1, `pd_joint_delta_pos`, physx_cpu, open-loop demo replay from recorded
  env_states, all eligible episodes (n≈993), paired per episode (same seed/state across arms).
- Arms: original, exact_integral (ZOH), cubic_spline_satfix, pchip_satfix, tac_fold_satfix, qp.
- k ∈ {2, 3, 4, 5}. Exact McNemar; paired bootstrap 95% CI.

## Predictions (committed now)
P1 (PRIMARY, headline lives or dies here): at k=4, qp beats cubic_spline_satfix by ≥ +3pp, and the
   contrast survives Holm over the primary family {qp vs cubic_spline_satfix at k=3,4,5} (m=3).
P2: qp beats ZOH at every k (Holm-significant) — expected, every satfix arm already does.
P3: at k=2, qp ≈ cubic_spline_satfix (|Δ| ≤ 2pp, n.s.) and qp < tac_fold_satfix (tac_fold_satfix
   had the lowest MSE at k=2).
P4: at k≥3, qp ≥ tac_fold_satfix, with the margin growing from k=3 to k=5.
P5: within each k, the MSE ranking (ksweep) predicts the success ranking. If not, that is reported as
   a failure of the proxy, not hidden.

## Secondary family (reported in full, Holm m=16)
qp vs {ZOH, cubic_spline_satfix, pchip_satfix, tac_fold_satfix} × k ∈ {2,3,4,5}.

## Not allowed after seeing results
No change to the QP objective/solver/tolerances, no k or n selection, no reference-arm swap. Any loss
(including qp < tac_fold_satfix at k=4) is reported as measured.
