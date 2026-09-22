# Preregistration: Newt × TAC-Fold on ManiSkill3-Panda20

Frozen before checkpoint evaluation on 2026-09-21 KST.

## Question

Does TAC-Fold preserve closed-loop task success under 2× temporal action decimation better than
cubic spline and smoothing B-spline reconstruction for a pretrained multitask policy?

## Policy and suite

- Policy: official `nicklashansen/newt` `soup-20M-default.pt` state checkpoint.
- Suite: the 20 tasks in `suite.json`, selected solely by a shared Franka/Panda 7-D
  `pd_ee_delta_pose` action contract. No task may be removed based on results.
- Observations: simulator state, matching the checkpoint.
- Episodes: 100 reset seeds per task (`0..99`), paired across arms; 2,000 task-episodes/arm.

## Frozen inference protocol

- Newt MPPI plan horizon: 8 steps. The checkpoint's one-step latent dynamics are rolled out for
  this inference horizon; no weights are changed.
- Execute the full eight-step plan, then replan. This creates the action-chunk deployment setting
  required by the rate-conversion question.
- Decimation factor: `k=2`, giving four coarse displacement blocks per plan.
- Continuous dimensions: first six end-effector delta-pose coordinates.
- Gripper: seventh coordinate, causal hold within each two-step block for every reconstructed arm.
- Policy proposals are clipped to `[-1, 1]` before coarsening. Reconstructed actions are clipped
  to `[-1, 1]` by the controller; no post-hoc satfix is used in the primary family.
- A task episode succeeds iff the environment success flag is true at the final step.
- Planning randomness is a pure function of `(task, episode seed, replan index)` and is identical
  across arms before their states diverge.

## Arms

1. `native`: execute the eight-step Newt mean plan unchanged.
2. `zoh`: sum each two-step block and split it equally over two steps.
3. `cubic`: SciPy's default not-a-knot cubic interpolation of cumulative block sums.
4. `bspline`: the repository's existing smoothing B-spline (`s=0.01`) on cumulative block sums.
5. `tac_fold`: existing Taut-Akima conservative fold on cumulative block sums.

## Confirmatory tests

- Primary A: TAC-Fold vs cubic, pooled paired exact McNemar over all 2,000 task-episodes.
- Primary B: TAC-Fold vs B-spline, same test.
- Holm correction across the two primary tests at family-wise alpha `0.05`.
- Report unadjusted effect sizes in percentage points and both discordant counts.
- Per-task contrasts and TAC-Fold vs ZOH/native are secondary; correct the 40 per-task
  TAC-vs-spline tests as one Holm family.
- Report macro-average success (equal weight per task) and micro-average success.

## Interpretation fixed in advance

- Suite win: TAC-Fold is ahead on both primary contrasts and both survive Holm correction.
- Mixed: exactly one primary contrast survives.
- Null: neither survives.
- Every task, including floors, ceilings, ties, and losses, remains in the main result table.
- A native macro success outside 60--80% is reported as measured and does not trigger task
  replacement or checkpoint tuning.

## Smoke and failure rules

- A one-task, two-episode smoke test is implementation validation only and is never pooled.
- If the official checkpoint cannot load or reproduce nontrivial native behavior, stop and report
  the incompatibility; do not tune interpolation parameters against evaluation outcomes.
- Environment/package fixes that do not inspect arm outcomes are allowed and logged.
