# Pre-registration — qp/qp_anchor vs raw spline/B-spline, ManiSkill PushT-v1 open-loop

Written 2026-09-16 21:23 KST, before any result from this run exists.

## Why this exists
The only PushT-v1 comparison against raw (un-satfixed) spline/B-spline so far is closed-loop,
n=100, k=2 (`RESULTS_QP_ANCHOR_CITABLE.md`'s "second task"): qp_anchor lost to raw spline
(-4.0pp, p=0.62, not sig) and to tac_fold_satfix (-2.0pp, p=0.79, not sig). n=100 closed-loop is
noisy (PickCube's own closed-loop n=400 needed real n to resolve a 3pp effect). PushT-v1 has 719
eligible open-loop demo episodes on disk (`/home/user/maniskill_data/pusht_rl.h5`, never
previously used for this comparison) — same paired-replay protocol as PickCube's n=993 result,
much higher power than n=100.

## Note on which "PushT" this is
This is ManiSkill's own `PushT-v1` (`harness.py:51`: `pd_ee_delta_pose` control, gripper-free,
raw demo actions reach |a|=3.3, i.e. genuinely delta/saturating), NOT lerobot's gym-pusht
(absolute-position control, used for the separate paper-reproduction track in
`PREREG_PUSHT_NATIVE.md`). Measured directly: raw single-step |a|>=0.999 fraction is 8.9% overall,
up to 15.7% in the worst dimension — real saturation, unlike gym-pusht.

## Protocol
- ManiSkill PushT-v1, `pd_ee_delta_pose`, physx_cpu, open-loop demo replay, n=719 (all eligible
  episodes — every episode in `pusht_rl.h5` has length >=16).
- Registry entry added to `sweep_maniskill_decimation_ratios.py::TASK_CONFIGS` (`has_gripper:
  False`, mirroring the existing PickCube/LiftPegUpright/PushCube/StackCube entries) — the only
  code change; `run_qp_replay.py` itself is untouched from the PickCube fair-raw run.
- Arms: original, exact_integral (ZOH), cubic_spline_satfix, tac_fold_satfix, bspline_eps_satfix,
  spline (raw), bspline_eps_raw, qp, qp_anchor.
- k in {2, 3, 4}.

## Predictions (committed now, before data)
P1 (PRIMARY): qp_anchor beats raw spline at every k, given PickCube's open-loop raw comparison
  (this session, same day) showed raw spline collapsing hard once un-boosted by satfix (45%/13%/6%
  at k=2/3/4). Genuinely uncertain here because PushT's closed-loop k=2 loss to raw spline is the
  one data point cutting the other way — this is the test that resolves the conflict.
P2 (PRIMARY): qp_anchor beats raw bspline_eps_raw at every k, same basis as PickCube (raw B-spline
  collapsed to 24%/11%/2% there).
P3: given the closed-loop n=100 loss to tac_fold_satfix, no directional prediction against
  tac_fold_satfix or cubic_spline_satfix — report as measured, not Holm-corrected as primary.

## Primary family for Holm correction (m=6)
qp_anchor vs {spline, bspline_eps_raw} x k in {2,3,4}.

## Not allowed after seeing results
No change to arm construction or k values. No dropping arms. A loss (including qp_anchor losing
to raw spline at any k, which would reverse-confirm the closed-loop result instead of resolving
it) is reported as measured, not re-run or reframed.
