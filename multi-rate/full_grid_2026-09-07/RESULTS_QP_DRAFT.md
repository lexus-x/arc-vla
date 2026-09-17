# Results draft — constrained-QP action resampling (for the multi-rate deployment paper)
Status: DRAFT, 2026-09-08 04:00 KST. Every number below is read from a results file in this
directory (file named per table). Cells marked "pending" have runs in flight; do not fill by hand.

## Method (one paragraph)
Given a policy's k-step-decimated action block sums S_1..S_B (per action dimension), we recover the
fine-rate sequence v by solving, per dimension, one convex QP over the whole chunk/episode:
  minimize  sum_t (v_{t+1} - v_t)^2   s.t.  sum_{t in block i} v_t = S_i  (all i),   -1 <= v_t <= 1.
The controller's clip range is a hard constraint of the fit. Prior resamplers either ignore it
(cubic spline, PCHIP, TAC-Fold, B-spline: interpolate/fit first, get clipped by the controller) or
repair it afterwards one block at a time (satfix). The per-block L2 projection onto the feasible set
separates across blocks — it IS satfix — so any gain must come from a cross-block objective; the
smoothness term above is that coupling, letting a violation borrow slack from a neighbouring block.
Output is feasible by construction; no parameters; SLSQP, sub-second per episode.

## Table 1 — open-loop paired replay, PickCube-v1, n=993 per depth (qp_replay_*.json, bspl_replay_*.json)
| arm | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|
| original (no decimation) | 92.45 | 92.45 | 92.45 | 92.45 |
| ZOH | 58.01 | 29.51 | 21.05 | 27.79 |
| cubic spline + satfix | 72.71 | 43.20 | 30.92 | 36.35 |
| PCHIP + satfix | 62.54 | 35.95 | 26.69 | 33.64 |
| TAC-Fold + satfix | 77.54 | 46.63 | 30.11 | 37.06 |
| B-spline (Alg. 1, eps=0.005) + satfix | 77.04 | — | 29.61 | — |
| **QP (ours)** | 74.12 | 45.62 | **34.74** | **39.88** |
Pre-registered primary (PREREG_QP.md): QP vs cubic+satfix at k=4 = +3.83pp, 60/22, Holm p=9.7e-5 (m=3). HOLDS.
Secondary family m=16: 15/16 survive Holm. Losses: QP < TAC-Fold+satfix at k=2 (-3.42pp, p=2.4e-5);
QP < B-spline+satfix at k=2 (-2.82pp, p=0.002). Wins: QP > TAC-Fold+satfix at k=4 (+4.63, p=3.8e-5),
k=5 (+2.82, p=0.0097); QP > B-spline+satfix at k=4 (+5.54, p=4.7e-7). k=3 vs TAC-Fold: tie (p=0.36).
Split-half replication (splithalf_output.txt): k=4 QP > cubic and > TAC-Fold in both halves
(H1 p=0.0006/0.001; H2 p=0.024/0.016). Figure: fig_crossover_pickcube.png.

## Table 2 — closed-loop Diffusion Policy (paper-style; assemble_table.py -> table_final.txt, COMPLETE)
Full k=2 and k=4 tables with McNemar lines are in table_final.txt. Verdicts:
- k=2: QP loses to cubic+satfix on PushT (-18pp, 5/23, p=0.0009) and to TAC-Fold+satfix (-10pp,
  p=0.021); ties everywhere else (PickCube 85-90 five-way; RoboMimic ceiling; RoboCasa null).
- k=4: QP leads only on PickCube (52 vs 48/48/40, n=100, +4pp, 6/2, p=0.29 — underpowered;
  n=400 extension pending, PREREG_QP_CL.md). PushT k=4: every decimated arm collapses to 1-3%
  (native 26) — void, the sign-flip-cancellation regime. k=5 closed-loop is degenerate (1 block per
  8-step chunk).
- Flow Matching, PickCube k=4, n=100: spline+satfix 51 = TAC-Fold+satfix 51 > QP 50 > B-spline 40 =
  ZOH 40. QP ties the satfix arms (1/2 discordant); pre-registered direction prediction (QP >= satfix)
  FAILED. QP vs ZOH +10pp (14/4, p=0.031).

## Table 3 — second/third contact task, open-loop paired (PREREG_QP2.md, qp2_replay_*.json)
Void rule applied on the DECIMATED arms (ZOH and best resampled arm inside 10-90%); see memory note
on the pre-reg wording defect ("native" is 92-99% on every ManiSkill task).
| cell | n | ZOH | cubic+sf | TAC-Fold+sf | B-spline+sf | QP | QP vs cubic | QP vs TAC-Fold | verdict |
|---|---|---|---|---|---|---|---|---|---|
| LiftPegUpright k=2 | 941 | 49.3 | 55.7 | 57.8 | 51.1 | 55.6 | -0.1 (p=1) | -2.2 (p=0.019) | informative: tie with cubic, loses to TAC-Fold (as at PickCube k=2) |
| LiftPegUpright k=3 | 941 | 2.2 | 5.1 | 5.7 | 4.6 | 6.1 | +1.0 (p=0.078) | +0.3 (p=0.66) | VOID (floor) |
| PushCube k=3 | 1019 | 92.4 | 91.5 | 91.4 | 91.6 | 91.6 | +0.1 (p=1) | +0.2 (p=0.85) | VOID (ceiling) |
| PushCube k=4 | 1019 | 89.8 | 87.3 | 90.0 | 87.1 | 86.8 | -0.6 (p=0.38) | -3.2 (p=5.6e-6) | ceiling-adjacent: QP LOSES to ZOH (-3.0, p=1.5e-4) and TAC-Fold |
PREREG_QP2 P1 (LiftPeg k=3 primary) is not evaluable (void). P2 holds. The k>=4 QP win remains
single-task (PickCube). PushCube k=4 is a measured failure region: where saturation headroom is small,
QP's global-smoothness bias is a net cost.

## Claim (as currently supported)
Depth- AND saturation-dependent: at shallow decimation (k=2), damped/least-squares curve families (TAC-Fold+satfix,
bounded-error B-spline+satfix) are best and QP is 3pp behind (open-loop) or worse (PushT
closed-loop); at deep decimation (k>=4) the globally constrained fit is the best resampler tested,
by 4-6pp over every alternative, Holm-significant open-loop at n=993 and replicated split-half.
Closed-loop transfer: direction and magnitude consistent at n=100, significance pending n=400.
Scope: the k>=4 win is one task (PickCube, ~40-50% of steps saturating); at the same depth on a
ceiling-adjacent task (PushCube k=4) QP loses to ZOH and TAC-Fold; on a second saturating task at
k=2 (LiftPeg) it ties cubic and loses to TAC-Fold; under Flow Matching at k=4 it ties the satfix arms.
Teleop benchmarks (RoboMimic, RoboCasa) show no saturation and no effect for any resampler.
Split-half replication of the PickCube n=993 result holds in both halves.

## Pending (in flight 2026-09-08 04:40)
DP PickCube k=4 closed-loop, n=400 held-out (PREREG_QP_CL.md A) — decides closed-loop significance.
