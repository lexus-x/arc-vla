# Pre-registration: DP seed replicates (written 2026-09-07 15:4x KST, before any training ran)

## What and why
`full_grid_2026-09-07` ran every cell at a single training seed (`torch.manual_seed(0)`), so no
result in this project separates a real interpolant effect from one draw of DP's initialisation +
batch order. This campaign adds seeds 1 and 2 for DP on the only two tasks where any effect has
ever reproduced (ManiSkill PushT-v1, PickCube-v1 -- saturating RL demos, FIRST_PRINCIPLES.md S3).
RoboMimic/RoboCasa are deliberately excluded: at ceiling (lift/can 100%) or floor
(PnPCounterToStove 6.7%), extra seeds there measure nothing.

## Protocol
Code copied frozen from `full_grid_2026-09-07` (dp_min.py byte-identical; harness.py differs only
by `--seed` plumbed into `torch/np` seeding and into the checkpoint/result tag). DP, 30k steps,
n_train=200, n_eval=100, k=2, all 10 arms, EMA weights, DDIM-10, paired McNemar vs zoh and vs
native. Eval init states are the same held-out demos as seed 0, so seeds are paired replicates.

## Predictions (fail = reported as failed, not reframed)
- P1: `spline_satfix` beats `zoh` on PushT-v1 in **all three** seeds (seed 0: +22.0pp, p=6e-05).
  If it holds in 1/3 or 2/3, the 09-07 result was seed-luck and must be reported as such.
- P2: `bspline` collapses on both tasks in all seeds (seed 0: 3% PushT, 44% PickCube). This is a
  numerical bug in `resample_bspline`, not a method property -- it should be seed-independent.
- P3: PickCube shows no arm beating `native` in any seed (seed 0: native 90% was the max).

## Not allowed after seeing results
No adding seeds until a prediction flips; no dropping either task; no changing arms, k, steps,
n_eval; no swapping the reference arm. Seeds 1 and 2 were fixed before launch. Pooling across
seeds, if done, is decided now: report per-seed tables first, pooled n=300 second, never only
pooled.
