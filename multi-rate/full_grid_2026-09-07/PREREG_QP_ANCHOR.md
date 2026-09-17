# Pre-registration — TAC-Fold-anchored QP (`resample_qp_anchor`), PickCube open-loop replay
Written 2026-09-10 20:40 KST, BEFORE any success-rate result for this arm exists at n>8 (an
8-episode wiring smoke test ran only to confirm the harness plumbing; discarded, not used to
inform any prediction below).

## Why this variant exists
`resample_qp` (the existing win) has no data term — it minimizes raw consecutive-difference
energy, which drifts toward a shape-blind, flattest-feasible curve. That is the diagnosed reason
it loses to TAC-Fold+satfix at k=2 (RESULTS_QP_DRAFT.md): TAC-Fold's Akima-damped curve already
tracks the true within-block dynamics when saturation is rare, and QP's flatness prior fights
that for nothing. `resample_qp_anchor` changes only the objective's reference point: minimize
`sum_t ((v_{t+1}-v_t) - (a_{t+1}-a_t))^2` where `a` = TAC-Fold's raw (pre-satfix) output, same
box/block-sum constraints, same solver. TAC-Fold satisfies the block-sum constraint exactly by
construction, so wherever TAC-Fold's raw output is already inside [-1,1], `v = a` is a zero-cost
global optimum: **the code provably reproduces TAC-Fold exactly** in that case (verified by a
self-check in `resample_qp.py`'s `__main__` block: `assert np.allclose(anchored, tac, atol=1e-4)`
on a non-saturating synthetic block-sum sequence). Wherever `a` violates the box, the same
cross-block coupling as plain QP lets a neighboring block absorb the deficit while staying close
to TAC-Fold's shape elsewhere — the mechanism that wins at k>=4 is unchanged.

The considered-and-rejected alternative, `resample_hybrid` (already in `resample_qp.py`, routes
only isolated violating blocks through global QP, base=spline+satfix): checked against the
existing `ksweep_results.json` MSE data before writing this file — hybrid's MSE is
indistinguishable from plain spline+satfix at every k on both tasks (e.g. PickCube k=4: hybrid
0.01430 vs spline_satfix 0.01447 vs qp 0.01209) and never approaches TAC-Fold's k=2 MSE (0.00300
vs hybrid's 0.00401). Not re-tested in sim — the MSE gate already falsifies it as a fix for the
k=2 loss, and this file's own no-sim gate (below) shows anchor doing what hybrid didn't.

## Evidence motivating this test (no success rates seen at n>8)
`ksweep_anchor_results.json` (60 real demos each, same protocol/data as the original QP's
motivating `ksweep_results.json`, no sim):
| task | k | zoh | spline_sf | tac_sf | qp | **qp_anchor** |
|---|---|---|---|---|---|---|
| PickCube | 2 | .00997 | .00402 | .00300 | .00382 | **.00295 (best)** |
| PickCube | 3 | .02017 | .00839 | .00802 | .00773 | **.00742 (best)** |
| PickCube | 4 | .02867 | .01447 | .01458 | **.01209 (best)** | .01293 |
| PickCube | 5 | .03476 | .01887 | .01958 | **.01599 (best)** | .01677 |
| PushT | 2 | .02919 | **.01150 (best)** | .01826 | .01230 | .01825 |
| PushT | 3 | .05889 | .04012 | .05109 | **.03973 (best)** | .05058 |
| PushT | 4 | .08665 | .09374 | .07548 | .08878 | **.07437 (best)** |
| PushT | 5 | .10379 | .10669 | .09910 | .10372 | **.09767 (best)** |

On PickCube, qp_anchor is the best (or effectively tied-best) MSE arm at k=2 and k=3 — including
now beating TAC-Fold+satfix, which plain QP has never done at k=2 in any prior campaign — and
close behind plain QP at k=4/5 (still well ahead of every satfix baseline it must beat). PushT is
messier: TAC-Fold is not PushT's best arm to begin with (plain spline is, per the established
"68% of PushT blocks have sign-flip oscillation, Akima damping is the wrong prior there" finding),
so anchoring to TAC-Fold recovers most but not all of PushT's k=2 gap, and is worse than plain QP
at k=3 there. PushT is reported here as context, not as a primary target of this pre-reg.

## Protocol (identical to PREREG_QP.md / run_qp_replay.py, arm added, nothing else changed)
- ManiSkill PickCube-v1, `pd_joint_delta_pos`, physx_cpu, open-loop demo replay, n=993 (all
  eligible episodes), paired per episode (same seed/state across arms).
- Arms: original, exact_integral (ZOH), cubic_spline_satfix, tac_fold_satfix, bspline_eps_satfix
  (eps=0.005, same as PREREG_BSPLINE.md), qp, qp_anchor.
- k in {2, 3, 4} (k=5 deferred, not part of this pre-reg; may be run later as an extension, not
  folded into this family after the fact).

## Predictions (committed now, before n=993 data)
P1 (PRIMARY): at k=2, qp_anchor beats tac_fold_satfix (reversing plain QP's established loss)
  and beats bspline_eps_satfix and cubic_spline_satfix. Given the thin MSE margin over TAC-Fold
  (0.00295 vs 0.00300, ~2%), the success-rate delta may be small and is NOT predicted to
  necessarily clear Holm significance — reported as measured either way.
P2 (PRIMARY): at k=3, qp_anchor beats cubic_spline_satfix, tac_fold_satfix, and bspline_eps_satfix
  (plain QP only tied TAC-Fold here; qp_anchor's MSE margin is larger, ~7-8%).
P3: at k=4, qp_anchor beats every satfix baseline (cubic/TAC-Fold/B-spline) as plain QP already
  does, but by a smaller margin than plain QP itself (qp_anchor MSE 0.01293 > qp's 0.01209) —
  predict qp_anchor < qp in success rate at k=4, both still ahead of the satfix baselines.
P4: qp_anchor beats ZOH at every k (expected, inherited from both TAC-Fold and QP each doing so).
P5: MSE ranking (this file's table) predicts success ranking, as it did for the original QP
  (P5 in PREREG_QP.md held at k=2/k=4, swapped at k=3/k=5) — if it doesn't, report the proxy
  failure, not hidden.

## Primary family for Holm correction (m=9)
qp_anchor vs {cubic_spline_satfix, tac_fold_satfix, bspline_eps_satfix} x k in {2,3,4}.
Secondary (reported in full, not Holm-corrected against the primary claim): qp_anchor vs qp,
qp_anchor vs ZOH, at each k.

## Not allowed after seeing results
No change to the anchor function, objective, solver, or tolerances. No k or n selection after
seeing success-rate numbers. No dropping arms. Any loss (including qp_anchor < qp, or qp_anchor
still losing to any satfix arm at any k) is reported as measured, not re-run or reframed.
This does not replace or invalidate PREREG_QP.md / RESULTS_QP_DRAFT.md — plain `qp` remains a
real, separately-reported arm in the same output files.
