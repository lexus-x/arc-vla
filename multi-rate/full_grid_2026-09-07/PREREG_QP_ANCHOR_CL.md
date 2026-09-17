# Pre-registration — QP-anchor, closed-loop DP and FM, PickCube-v1 k=4
Written 2026-09-10 21:15 KST, BEFORE any closed-loop success-rate result for qp_anchor exists.

## Basis (open-loop, n=993, already measured and reported, PREREG_QP_ANCHOR.md)
qp_anchor beats cubic_spline_satfix, tac_fold_satfix, bspline_eps_satfix at k=4 open-loop
(+2.5/+3.4/+3.7pp, all Holm-sig, m=9); by a SMALLER margin than plain qp (qp's own k=4 margins
were +4.0/+4.9/+5.2pp) as predicted in P3. Plain qp's own closed-loop transfer at k=4 (DP,
PREREG_QP_CL.md A) was direction-consistent but not significant at n=400 (+2.0pp, p=0.15) --
the open-loop-to-closed-loop shrinkage (open-loop +3.8pp -> closed-loop +2.0pp for plain qp
vs cubic_satfix) is the established pattern here, not new.

## Predictions (committed now)
P1: qp_anchor at closed-loop DP k=4 (n=400, same protocol/checkpoint as PREREG_QP_CL.md A,
   dp_PickCube-v1.pt) leads spline_satfix/tac_fold_satfix/bspline_eps_satfix in direction, by a
   MARGIN NO LARGER than plain qp's own +2.0pp (open-loop already showed qp_anchor <= qp at k=4)
   -- given plain qp's own +2.0pp missed significance at n=400, qp_anchor's closed-loop margin is
   NOT expected to reach significance either. This is stated in advance as a likely-negative
   prediction, not hidden after the fact.
P2: qp_anchor at closed-loop FM k=4 (n=100, fm_PickCube-v1.pt) reverses plain qp's tie (qp tied
   satfix arms at 50 vs 51, PREREG_QP_CL.md B's "qp>=satfix" direction FAILED) to a real lead,
   given qp_anchor's larger open-loop margin over bspline/tac_fold at k=4 than plain qp's FM
   result showed room for -- reported as measured regardless.
P3: qp_anchor beats ZOH at both (expected, inherited).

## Protocol
Identical to PREREG_QP_CL.md / harness.py: same checkpoints (no retraining -- `os.path.exists(ckpt)`
load-path only), same k=4, same n (400 DP / 100 FM), arms native,zoh,spline_satfix,
tac_fold_satfix,bspline_eps_satfix,qp,qp_anchor. Holm m=3 per policy (qp_anchor vs the 3 satfix
arms), as PREREG_QP_CL.md A used.

## Not allowed after seeing results
No retraining, no n change, no dropping arms, no swapping k. A non-significant or negative P1 is
reported exactly as predicted, not reframed.
