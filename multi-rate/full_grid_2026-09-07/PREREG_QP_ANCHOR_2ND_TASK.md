# Pre-registration — qp_anchor, second closed-loop task: PushT, k=2

Written 2026-09-15 00:05 KST, before any qp_anchor success-rate result exists for PushT.

## Why this task, not a "safer" one

RESULTS_QP_ANCHOR_CITABLE.md's PickCube win (n=400, k=4) is single-task. Candidates for a second
task were assessed by what this repo's OWN existing data already shows about each, not chosen
blind:
- Lift/Can/Square (`table_final.txt`): base SR 89-100% — near-ceiling, almost no room for a
  resampler to differ (the k=2 table's own McNemar lines are already mostly 0/0 or 1-vs-1
  discordant pairs). Expected: VOID, like `PREREG_QP2`'s ceiling cells.
- RoboCasa tasks: established finding across this project — "no saturation, no effect for any
  resampler" at native rate.
- PushCube / LiftPegUpright (`RESULTS_QP_DRAFT.md` Table 3): genuine non-ceiling dynamic range,
  but no closed-loop DP checkpoint exists for either — would require training from scratch first
  (~4-6h), deferred as a possible third task, not this one.
- **PushT k=2** (`table_final.txt`): checkpoint already exists (`dp_PushT-v1.pt`, no retraining —
  `os.path.exists(ckpt)` load-path reuse, verified in `harness.py:283`), and plain `qp` **already
  loses** here: vs spline_satfix -18pp (5/23, p=0.00091), vs tac_fold_satfix -10pp (3/13, p=0.021),
  both significant at only n=100. This is the sharper test: does anchoring to TAC-Fold's own shape
  fix a known QP failure, or is the PickCube win idiosyncratic? Either outcome is informative and
  reported as measured.

## Hypothesis

`qp_anchor`'s TAC-Fold-anchored objective should recover most of plain QP's PushT k=2 loss to
TAC-Fold (by construction, `qp_anchor` reproduces TAC-Fold exactly wherever TAC-Fold's own output
is feasible — see `resample_qp.py::resample_qp_anchor` docstring) but is NOT expected to fully
close the gap to cubic-spline+satfix, which the project's own prior finding attributes to PushT's
"sign-flip oscillation" regime (≈68% of blocks) where even TAC-Fold's Akima damping is not the
best-suited prior (`PREREG_QP_ANCHOR.md`: "PushT is messier... anchoring to TAC-Fold recovers most
but not all of PushT's k=2 gap").

## Protocol

**Amended 2026-09-15 10:50 KST, before any result for this run existed (the prior invocation was
killed pre-completion, no data seen) — house rule: satfix is our own post-hoc repair mechanism;
lending it to competing baselines (cubic spline, B-spline/paper's Alg.1) overstates their
performance. satfix is now used only where it always was ours to use: wrapping `tac_fold`. `qp`/
`qp_anchor` never used satfix to begin with (feasible by construction). `spline` and
`bspline_eps_raw` are now compared in their raw, un-repaired form.**

`python harness.py PushT-v1 --k 2 --seed 0 --arms native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor`
— same checkpoint (`dp_PushT-v1.pt`, loaded not retrained), same n=100 as the existing table cell
(already shown sufficient to detect this size of effect), one combined run (all arms paired per
episode in a single invocation, not spliced across separate runs, to avoid any seed/pairing drift).

## Predictions (committed now)

P1 (PRIMARY): qp_anchor vs tac_fold_satfix — direction reverses plain QP's loss (positive or near-
zero delta), given qp_anchor reproduces TAC-Fold wherever feasible.
P2 (PRIMARY): qp_anchor vs raw spline (no satfix) — expected margin is now LARGER than the old
-18pp-vs-spline_satfix comparison, since raw spline is no longer getting our repair; still reported
as measured, not assumed.
P3: qp_anchor vs zoh — beats it (inherited from TAC-Fold and QP both doing so at k=2 elsewhere).
P4: qp_anchor vs raw bspline_eps (no satfix, eps=0.005, the paper's own Alg.1) — no strong prior;
reported as measured.

## Primary family for Holm correction (m=2)

qp_anchor vs {tac_fold_satfix, spline (raw)}. Secondary (reported, not Holm-corrected against the
primary claim): qp_anchor vs zoh, qp_anchor vs bspline_eps_raw, qp_anchor vs qp, qp_anchor vs
native.

## Not allowed after seeing results

No changing k, n, arms, or checkpoint after seeing results. No re-running if P1/P2 fail — a
continued loss to spline_satfix, or a failure to fix the tac_fold_satfix loss, is reported exactly
as measured, and is itself the honest answer to "does the mechanism generalize."
