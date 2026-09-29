# Training-free resampler results

Completed: 16/16 task/rate cells. Fresh paired batched closed-loop rollouts.
Fixed seed-0 DP checkpoints; no resampler training. B-spline eps=.005.
This saturation-screened ManiSkill suite does not establish a universal winner.

| Task | k | n | native | zoh | tac_fold | spline | bspline_eps_raw | tac_fold_satfix | spline_satfix | bspline_eps_satfix | qp | qp_anchor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PullCube-v1 | 2 | 400 | 96.50% | 97.50% | 94.75% | 89.25% | 86.25% | 97.00% | 97.00% | 97.00% | 97.50% | 97.75% |
| PullCube-v1 | 4 | 400 | 96.50% | 47.50% | 36.75% | 37.25% | 47.50% | 51.00% | 51.75% | 47.50% | 54.50% | 58.00% |
| PickCube-v1 | 2 | 293 | 82.94% | 72.35% | 76.45% | 66.55% | 55.63% | 81.23% | 80.55% | 80.55% | 80.20% | 81.57% |
| PickCube-v1 | 4 | 293 | 82.94% | 36.86% | 26.28% | 26.28% | 36.86% | 44.37% | 44.37% | 36.86% | 46.08% | 45.05% |
| PushCube-v1 | 2 | 400 | 96.75% | 94.75% | 93.25% | 92.00% | 94.25% | 94.75% | 93.75% | 96.75% | 95.25% | 93.75% |
| PushCube-v1 | 4 | 400 | 96.75% | 69.00% | 57.75% | 57.75% | 69.00% | 66.50% | 65.25% | 69.00% | 66.25% | 65.25% |
| PokeCube-v1 | 2 | 400 | 95.00% | 94.50% | 94.25% | 94.00% | 93.75% | 93.25% | 94.50% | 93.75% | 94.75% | 93.50% |
| PokeCube-v1 | 4 | 400 | 95.00% | 79.75% | 79.25% | 79.75% | 79.75% | 81.25% | 81.00% | 79.75% | 82.00% | 81.50% |
| LiftPegUpright-v1 | 2 | 400 | 95.25% | 49.75% | 45.00% | 13.50% | 5.50% | 58.75% | 60.75% | 57.75% | 60.75% | 61.00% |
| LiftPegUpright-v1 | 4 | 400 | 95.25% | 0.50% | 1.75% | 2.00% | 0.50% | 3.00% | 2.75% | 0.50% | 2.25% | 4.50% |
| StackCube-v1 | 2 | 400 | 35.50% | 23.25% | 27.50% | 22.00% | 13.00% | 29.25% | 27.50% | 24.75% | 25.25% | 29.50% |
| StackCube-v1 | 4 | 400 | 35.50% | 2.50% | 1.50% | 1.50% | 2.50% | 7.00% | 6.75% | 2.50% | 9.25% | 7.25% |
| RollBall-v1 | 2 | 400 | 61.00% | 35.50% | 45.50% | 36.00% | 20.75% | 43.25% | 45.00% | 44.25% | 42.00% | 43.50% |
| RollBall-v1 | 4 | 400 | 61.00% | 9.00% | 9.25% | 9.25% | 9.00% | 13.75% | 13.50% | 9.00% | 13.50% | 16.00% |
| AnymalC-Reach-v1 | 2 | 400 | 92.75% | 63.25% | 67.75% | 65.25% | 45.50% | 71.25% | 70.50% | 71.25% | 69.00% | 77.00% |
| AnymalC-Reach-v1 | 4 | 400 | 92.75% | 1.50% | 6.25% | 5.00% | 1.50% | 4.50% | 6.50% | 1.50% | 7.50% | 12.00% |

Equal-weight mean across all 16 cells (descriptive):

- native: 81.96%
- qp_anchor: 54.20%
- qp: 52.88%
- spline_satfix: 52.59%
- tac_fold_satfix: 52.51%
- bspline_eps_satfix: 50.79%
- zoh: 48.59%
- tac_fold: 47.70%
- spline: 43.58%
- bspline_eps_raw: 41.33%

Exact paired McNemar with Holm correction: contrasts.json (separate 48-test raw/SATFix families).
