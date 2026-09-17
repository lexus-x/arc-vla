# Pre-registration — qp_anchor, 3rd task: RoboCasa CloseSingleDoor (paper's own benchmark task)

Written 2026-09-15 11:53 KST, before any result exists. Compact by necessity (time-constrained) but
still committed before data.

## Why this task
CloseSingleDoor is one of the paper's (arXiv 2607.09648) own 6 Table 2(a) tasks (Diff. base 27%,
Reg. base 40%) — unlike PickCube (not in the paper at all), a result here is directly
paper-comparable in kind, not just in mechanism. Existing checkpoint `dp_RC-CloseSingleDoor_final.pt`
(complete, no retraining), bridge already running (port 8765, 5+ days uptime).

## Protocol
`python harness.py RC-CloseSingleDoor --k 2 --n_train 39 --n_eval 15 --seed 0 --workers 4 --arms native,zoh,spline,bspline_eps_raw,tac_fold_satfix,qp,qp_anchor`
— satfix restricted to tac_fold only (standing rule), raw spline/bspline_eps otherwise. n=15 is
small (this task's established held-out set, matches `table_final.txt`'s existing RC convention) —
likely underpowered for significance; reported as measured regardless, per house rule.

## Predictions
No strong prior for k=2 on this specific task (first time qp_anchor is tested on RoboCasa). Direction
expectations only, inherited from the general mechanism: qp_anchor >= plain qp; qp_anchor beats zoh.
No prediction on spline/tac_fold_satfix given n=15's low power — reported as measured either way.

## Primary family (Holm m=2)
qp_anchor vs {tac_fold_satfix, spline (raw)}.

## Not allowed
No re-running after seeing results. n=15 underpowered is stated in advance, not discovered after a
null result. No cherry-picking a different task/checkpoint if this one disappoints.
