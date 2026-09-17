# Pre-registration — QP vs B-spline (as a resampler), PickCube open-loop paired replay, n=993
Written 2026-09-08 00:50 KST, BEFORE any success-rate result for a B-spline arm on this harness.

## Why this comparison and not "their Table 2 vs ours"
The B-spline Policy paper (arXiv 2607.09648) trains the policy to emit spline parameters (16 knots +
control points); it is a policy OUTPUT REPRESENTATION, evaluated on their own 200 Hz image-based
Push-T variant plus RoboMimic Lift and four RoboCasa tasks we do not have as state-based datasets.
Their numbers and ours are not on the same axis or the same environments and will not be compared.
What CAN be compared on one axis, one dataset, one policy-free harness: their curve-fitting
algorithm (Alg. 1, cubic B-spline with adaptive knot insertion under a max-error tolerance eps)
used as a post-hoc chunk resampler, against our QP resampler. This also retests the campaign's
"B-spline collapses" cell with a numerically sound implementation (`resample_bspline2.py`),
replacing the splprep(s=0.01) arm that blew up.

## Facts established before this replay (reconstruction only, no success rates)
- A cubic B-spline INTERPOLANT (make_interp_spline, k=3) is identical to the cubic spline arm to
  5 digits in every cell: same function space, same not-a-knot boundary. So "QP vs B-spline
  interpolation" is already answered by qp_replay (QP > cubic_spline_satfix, Holm-sig at k>=3).
- The bounded-error (least-squares) B-spline, Alg. 1 with midpoint/bisection knot insertion,
  knot cap n-5, evaluation-grid validity guard: converges (knots increase, MSE falls) as eps -> 0.
  Best eps = 0.005 (monotone; pre-chosen here). satfixed MSE vs QP: PickCube k=2 0.00596 vs
  0.00343; k=4 0.01483 vs 0.01141; PushT k=2 0.01729 vs 0.01143. Worse than cubic_spline_satfix
  everywhere too.

## Arms (k in {2, 4}, same physx_cpu replay, exact McNemar, paired)
original, exact_integral (ZOH), cubic_spline_satfix, tac_fold_satfix, bspline_eps_satfix (eps=0.005), qp.

## Predictions
P1 (primary): qp > bspline_eps_satfix at BOTH k=2 and k=4, each surviving Holm over m=2.
P2: bspline_eps_satfix > ZOH at both k (satfix restores sums; shape is still better than uniform).
P3: bspline_eps_satfix < cubic_spline_satfix at both k (least-squares smoothing loses within-block
    shape that the interpolant keeps).
P4: The campaign's "B-spline 44% at k=2 closed-loop / catastrophic" does NOT reproduce as a
    collapse here: bspline_eps_satfix at k=2 lands between ZOH (58%) and cubic_satfix (72.7%).

## Not allowed after seeing results
No eps retuning, no knot-cap change, no arm/k/n change. Losses reported as measured.
