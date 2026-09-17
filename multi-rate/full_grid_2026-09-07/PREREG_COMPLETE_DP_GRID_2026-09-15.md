# Pre-registration — complete diffusion-policy rate grid

Written 2026-09-15 KST after inventorying existing files and before any full missing-cell run.
The user requested every missing measured cell at 1x/2x/4x/8x; no values may be inferred.

## Fixed scope

- Step-head DP: PushT-v1 and RoboMimic lift/can/square, seed 0, 200 train demos,
  100 paired evaluation episodes, existing 30k-step checkpoints.
- Rates: k in {1,2,4,8}.
- Arms: native, zoh, raw cubic spline, raw B-spline eps=.005,
  TAC-Fold+satfix, plain QP, and QP-anchor.
- Run only arms absent from an existing standard n=100 result. Noise, smoke, and validation
  files do not fill standard cells.
- Trained B-spline-head checkpoint: independently evaluate k in {2,4,8}; k=1 already exists.
  Report this execute-faster experiment separately from step-head multi-rate reconstruction.
- k=8 step-head is diagnostic/degenerate because the eight-action execution chunk contains one
  block; report it, but do not use it as evidence for QP cross-block coupling.

## Statistics and reporting

- Success rate and paired per-episode outcomes are read only from result JSONs.
- Exact paired McNemar for stated pairwise comparisons; Holm correction within each declared
  task/rate comparison family.
- No arm, task, or rate is dropped after observing results. Failures and floors remain explicit.
- Existing and new files are mapped to every final table cell; no hand-filled values.

## RoboCasa boundary

The same state-only grid is a separate low-data experiment (39 train/15 evaluation demos).
For the four paper tasks, evaluate all three pre-existing held-out folds independently at
k in {1,2,4,8}; step-head arms match the seven-arm list above, and the pre-existing trained
B-spline head is evaluated separately at k in {2,4,8}. For legacy OpenDrawer and
PnPCounterToStove, evaluate the seven step-head arms at all four rates; no trained B-spline-head
checkpoint exists, so those head cells are not created by an evaluation-only campaign.
Current full-scale RoboCasa image policies are not compatible with the state harness and predict
absolute actions, so the delta-action resamplers cannot truthfully be applied without a new,
pre-registered evaluator/semantics. Full-scale cells remain pending until that integration exists.

RoboCasa uses one sequential harness client per bridge. Four paper tasks run concurrently on four
independent bridge ports; folds and rates remain sequential within a task. Aggregate reporting may
pool the three fixed folds (n=45) while retaining every fold-level outcome.
