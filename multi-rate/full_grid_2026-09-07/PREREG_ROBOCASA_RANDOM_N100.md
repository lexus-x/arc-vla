# Pre-registration — RoboCasa random-reset closed-loop n=100

Written before any random-reset result exists. The existing state-only protocol is limited to
15 held-out demonstration states because each task has 54 demos (39 train/15 eval). This campaign
keeps the existing fold-0 checkpoint, training set, normalizers, and training budget unchanged,
but evaluates 100 deterministic fresh environment resets instead of recycling held-out demos.

## Fixed protocol

- Tasks: RC-TurnOffSinkFaucet, RC-CoffeePressButton, RC-TurnOffMicrowave,
  RC-CloseSingleDoor.
- Policy: existing fold-0 state-only Diffusion Policy checkpoint; no retraining.
- Evaluation: n=100 random resets with seeds 1,000,000 through 1,000,099.
- Rates: k in {1,2,4,8}.
- Paired step-head arms: native, zoh, raw spline, raw B-spline eps=.005,
  TAC-Fold+satfix, QP, QP-anchor.
- Separately evaluate each existing fold-0 trained B-spline-head checkpoint at all four rates.
  These are execute-faster results and must not be merged with the fixed-wall-time step-head table.
- One bridge/task, one sequential client; no unvalidated RoboCasa multiprocessing.
- Exact McNemar on paired episode outcomes. Report all cells, including floors and ties.
- k=8 is diagnostic only because the eight-action chunk contains one block.

## Interpretation boundary

This is an n=100 evaluation of the existing 39-demo state-only policy, not a paper-scale
reproduction. It can estimate that policy's success rate more precisely but cannot repair its
low native competence. It must not be mixed with 3,000-demo visual-policy claims.
