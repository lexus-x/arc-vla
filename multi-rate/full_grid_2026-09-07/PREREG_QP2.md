# Pre-registration — QP resampler on a SECOND and THIRD contact task (open-loop paired replay)
Written 2026-09-08 03:55 KST, before any result on these tasks for the QP or B-spline arm.

Limitation being addressed: the pre-registered k>=3 win (PREREG_QP.md) is PickCube only.
Tasks/depths (same physx_cpu replay harness, `run_qp_replay.py`, all eligible episodes):
- LiftPegUpright-v1 k=2 (satfix baseline known: n=941, cubic_satfix +5.6pp vs ZOH) and k=3
  (untested depth; k=4 is floor-collapsed per vault, ZOH 0.32% — NOT run, would be void).
- PushCube-v1 k=3 and k=4 (k=2 is ceiling per vault, all arms tie — NOT run, would be void).
Arms: original, exact_integral (ZOH), cubic_spline_satfix, tac_fold_satfix, bspline_eps_satfix (eps=0.005), qp.

Predictions:
P1 (primary): on LiftPegUpright k=3, qp >= cubic_spline_satfix (delta >= 0, McNemar p reported);
   a WIN is predicted only if LiftPeg's raw-spline violation rate at k=3 is >= 30% (measured in the
   run log as saturated_elems / checked post hoc from the data) — the mechanism says the edge scales
   with saturation frequency; if LiftPeg saturates little at k=3, expect a tie, not a win.
P2: on LiftPegUpright k=2, qp <= tac_fold_satfix (as on PickCube k=2) and qp ~ cubic_satfix.
P3: PushCube k=3/k=4: if the cell is not ceiling/floor (native and ZOH inside 10-90%), qp >= cubic_satfix;
   if ceiling/floor, report VOID, no contrast claimed.
Family for Holm: qp vs cubic_spline_satfix and qp vs tac_fold_satfix over the non-void cells (m <= 8).
Not allowed after results: dropping a non-void cell, changing eps/solver, adding k values.
