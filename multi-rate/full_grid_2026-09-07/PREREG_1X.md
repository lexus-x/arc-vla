# Pre-registration — 1X (no-speedup) smoothing + table completion vs B-spline Policy (arXiv 2607.09648) Table 2
Written 2026-09-08 12:35 KST. Already existing: DP PushT/Lift k=1 post-hoc B-spline (PushT +4/+8pp, n.s.), all DP k=2 cells, PickCube k=4.
Target: paper-format table (Base / +method) at 1X, 2X, 4X for DP and FM on PushT, Lift, PickCube, Can, Square.
New arm qp_eps (resample_qp, eps>0): min sum (v_{t+1}-v_t)^2 s.t. |cumulative block-sum error| <= eps at every
block end, |v| <= 1. eps=0 recovers the pre-registered QP unchanged. At 1X it is a bounded-position-error smoother
(our analog of the paper's Alg. 1 eps-fit, with the controller clip built in). eps in {0.05 (primary), 0.10}.
Runs: DP k=1 PickCube/Can/Square (native, zoh, bspline_eps_raw, bspline_eps05_raw, qp_eps05, qp_eps10);
DP k=1 PushT/Lift (native, qp_eps05, qp_eps10; merged with the existing k=1 file — pairing is deterministic per episode);
DP k=4 Lift/Can/Square (native, zoh, spline_satfix, tac_fold_satfix, qp); FM k=2 PushT/PickCube/Lift/Can/Square (native, zoh, tac_fold_satfix, qp).
Predictions: P1 (primary) DP PushT 1X: qp_eps05 > native (jittery policy, raw |a|>1 = 7%). P2: 1X smoothers tie native on
Lift/Can (ceiling) and PickCube/Square (|delta| <= 5pp). P3: DP 4X RoboMimic: qp >= tac_fold_satfix >= zoh, all far below native.
P4: FM 2X: qp ~ tac_fold_satfix. Holm family: qp_eps05 vs native over the 5 DP 1X tasks (m=5).
Not allowed: choosing eps after results, dropping cells, changing n. RoboCasa not run (n=15 floor collapse, bridge).
Caveat: paper PushT = 2D coverage score (base 72%); ours = ManiSkill PushT-v1 binary success (base 26%). Compare deltas, not levels.
