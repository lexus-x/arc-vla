# Complete diffusion-policy grid — measured snapshot

Updated 2026-09-15 after stopping non-citable compute. Success rate (%), n=100 paired closed-loop episodes.
RoboMimic cells are Lift/Can/Square. No values are inferred.

## Step-head policy with post-hoc resampling

| Suite | Rate | Native | QP | QP-anchor | TAC-Fold+satfix | Spline raw | B-spline raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| PushT | 1x | 26 | 26 | 26 | 27 | 30 | 30 |
| PushT | 2x | 26 | 20 | 28 | 30 | 32 | 11 |
| PushT | 4x | 26 | 3 | 2 | 2 | 4 | 2 |
| PushT | 8x | 26 | 0 | 0 | 0 | 0 | 0 |
| RoboMimic | 1x | 100/98/89 | 100/98/89 | 100/98/89 | 100/98/90 | 100/98/90 | 100/98/88 |
| RoboMimic | 2x | 100/98/89 | 100/97/90 | 100/97/86 | 100/98/88 | 100/98/84 | 100/98/87 |
| RoboMimic | 4x | 100/98/89 | 100/97/87 | 100/99/84 | 100/98/84 | 100/99/87 | 100/98/85 |
| RoboMimic | 8x | 100/98/89 | 99/80/79 | 99/80/79 | 99/80/79 | 99/80/79 | 99/80/79 |

## Trained B-spline Policy (execute-faster; separate semantics)

| Suite | 1x | 2x | 4x | 8x |
|---|---:|---:|---:|---:|
| PushT | 28+/-7 | 4 | 0 | 0 |
| RoboMimic | 100/99/94 | 100/35/58 | 85/15/14 | 13/0/4 |

## Compute verdict

- PushT/RoboMimic grid: complete and useful; 27/27 new result files have n=100 arrays.
- Step-head k=8: diagnostic only. One eight-action block removes QP cross-block coupling; do not repeat across seeds.
- Trained B-spline-head rates are execute-faster, not the same intervention as step-head multi-rate reconstruction.
- Low-data RoboCasa sweep was cancelled before evaluation: established zero-saturation/null regime.
- Four full-scale RoboCasa image trainings were stopped after epoch-200 checkpoints. They use absolute actions and cannot feed the current delta-action resamplers without a new evaluator and method definition.
