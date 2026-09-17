# Pre-registration — PushT-v1 closed-loop, fair arms, n=400 (power extension)

Written 2026-09-16 21:26 KST, before any result from this run exists.

## Why
The only closed-loop fair-arm PushT-v1 result is n=100, k=2 (`result_dp_PushT-v1_anchor_2ndtask_v2.json`,
reported in `RESULTS_QP_ANCHOR_CITABLE.md`'s "second task"): qp_anchor lost to raw spline (-4pp,
p=0.62) and tac_fold_satfix (-2pp, p=0.79) — both not significant, i.e. underpowered, exactly like
PickCube's own n=100 closed-loop cell was before its n=400 extension (`PREREG_QP_ANCHOR_CL.md`)
resolved it. Same extension here, same checkpoint, same policy, no retraining.

## Protocol
- `dp_PushT-v1.pt` (train_steps=30000, the exact checkpoint behind the existing n=100 result —
  confirmed by checkpoint-path construction in `harness.py:406`; not retrained).
- `--n_train 200` (default, unchanged — matches the checkpoint's own training split).
- `--n_eval 400`, eval episodes = eligible[200:600] of PushT-v1's 719 total. This *extends* (does
  not replace) the original eligible[200:300] n=100 window — same held-out region, more of it.
- k in {2, 3}. k=4 excluded: `RESULTS_QP_DRAFT.md` already documents closed-loop PushT k=4 as void
  (every decimated arm collapses to 1-3%, native 26% — the sign-flip-cancellation regime), so a
  larger n there would not be informative.
- Arms: native, zoh, spline (raw), bspline_eps_raw, tac_fold_satfix, qp, qp_anchor — same fair set
  as the n=100 run (satfix withheld from spline/B-spline, per the house rule already applied there).

## Predictions
P1 (PRIMARY): the k=2 direction (qp_anchor losing to raw spline and to tac_fold_satfix) either
  holds and reaches significance, or reverses. No directional prediction — the n=100 result was
  underpowered both ways; this is a resolution, not a directional bet.
P2: k=3 is new (never run for PushT-v1 closed-loop before); no prior direction to anchor to.

## Primary family for Holm correction (m=4)
qp_anchor vs {spline, tac_fold_satfix} x k in {2,3}.

## Not allowed after seeing results
No change to eval window, checkpoint, or arm list. No dropping k=3 if it's inconvenient. Report
exactly as measured, including a confirmed loss.
