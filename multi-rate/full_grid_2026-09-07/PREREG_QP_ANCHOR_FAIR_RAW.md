# Pre-registration — qp/qp_anchor vs the raw (un-satfixed) competing algorithms, PickCube open-loop

Written 2026-09-16 20:20 KST, before any result from this run exists. `run_qp_replay.py` is
being extended right now to add two arms it has never produced: `spline` and `bspline_eps_raw`,
un-repaired by our own `satfix` mechanism (previously only `cubic_spline_satfix` and
`bspline_eps_satfix` existed in this script — see `RESULTS_QP_ANCHOR_CITABLE.md`'s own gap note
and this session's audit that found no raw-arm PickCube comparison had ever been run).

## Why this exists
`PREREG_QP_ANCHOR_2ND_TASK.md:42`'s house rule — satfix is our own repair mechanism; lending it
to competing baselines overstates them — was applied at PushT but never at PickCube. Every prior
PickCube open-loop table (`RESULTS_QP_DRAFT.md` Table 1, `RESULTS_QP_ANCHOR_CITABLE.md`'s citable
table) compares qp/qp_anchor against `cubic_spline_satfix` and `bspline_eps_satfix` only. This is
the first fair (raw vs raw) comparison at PickCube.

## Protocol (identical harness, two new arms, nothing else changed)
- ManiSkill PickCube-v1, `pd_joint_delta_pos`, physx_cpu, open-loop demo replay, n=993 (all
  eligible episodes, same set as every prior PickCube open-loop table), paired per episode.
- Arms: original, exact_integral (ZOH), cubic_spline_satfix, tac_fold_satfix, bspline_eps_satfix,
  **spline (raw)**, **bspline_eps_raw**, qp, qp_anchor.
- k in {2, 3, 4} (same as `PREREG_QP_ANCHOR.md`).
- Raw arms are clipped at step time only (`np.clip(act,-1,1)` before `env.step`, matching
  `harness.py:135`'s treatment of raw arms elsewhere) — no block-sum-preserving redistribution.

## Predictions (committed now, before data)
P1 (PRIMARY): qp_anchor beats raw `spline` at every k. Basis: PushT closed-loop already showed
  raw spline is competitive there (28% vs 32%, a loss) — this is genuinely uncertain, not assumed
  to hold at PickCube; reported either way.
P2 (PRIMARY): qp_anchor beats raw `bspline_eps_raw` at every k, by a wide margin. Basis: PushT's
  raw B-spline collapsed to 11% vs its satfixed 27% — satfix was carrying most of its apparent
  competitiveness there; expect the same pattern here (bspline_eps_satfix is currently ~77/43/30%
  across k=2/3/4 — raw should be markedly lower).
P3: plain qp also beats both raw arms at every k (weaker prediction than P1/P2 since qp already
  loses to the satfixed versions at k=2).

## Primary family for Holm correction (m=6)
qp_anchor vs {spline, bspline_eps_raw} x k in {2,3,4}.
Secondary (reported, not Holm-corrected against the primary): qp vs {spline, bspline_eps_raw} at
each k; qp_anchor/qp vs the existing satfix arms (already reported previously, re-stated here for
context only, not re-tested as new).

## Not allowed after seeing results
No change to arm construction, clip behavior, or k values. No dropping arms. Any loss (including
qp_anchor losing to raw spline or raw bspline_eps at any k) is reported as measured.
