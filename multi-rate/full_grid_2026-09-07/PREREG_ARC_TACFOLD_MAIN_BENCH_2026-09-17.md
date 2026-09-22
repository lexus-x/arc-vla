# Pre-registration — ARC + TAC-Fold main simulator benchmark

Written 2026-09-17 before any ARC checkpoint or ARC result file existed.

## Method fixed before results

ARC (Action-Resolution Conditioning) is one diffusion-policy checkpoint trained jointly at action
resolutions `k={1,2,4}`. The scalar
`log2(k)/2` is concatenated to the observation condition. At each rate the policy predicts eight
block means, each covering `k` native demonstration actions. TAC-Fold converts those means to exact
block sums, reconstructs controller-rate actions, and executes the first eight controller steps.
Thus the rate changes both the conditioning input and supervised target; it is not an inert token.
This intervention tests action-resolution transfer, not wall-clock task acceleration.

The primary comparator is the same step-head diffusion backbone trained only on native action
chunks, then decimated and reconstructed post hoc by TAC-Fold+satfix. Training demonstrations,
optimizer, training steps, observation state, controller, evaluation initial states, policy noise,
and episode budget are matched. `arc+zoh` isolates the learned rate-conditioned representation;
`arc+tac_fold` versus `arc+zoh` isolates the decoder.

The named external representation baselines are raw cubic-spline reconstruction and the bounded-
error B-spline reconstruction (`bspline_eps_raw`) implemented from B-spline Policy Algorithm 1.
They do not receive the proposed satfix projection. Both are evaluated on the same predicted block
summaries and paired rollouts. This distinguishes the ARC representation effect from the choice of
continuous decoder without attributing our constraint repair to a competing method.

## Fixed benchmarks

- Push-T: `PushT-v1`, 200 training demonstrations, 400 paired closed-loop evaluation episodes.
- RoboMimic: Lift, Can, Square, 200 training demonstrations and 100 paired episodes per task.
- RoboCasa: TurnOffSinkFaucet, CoffeePressButton, TurnOffMicrowave, CloseSingleDoor; 39 training
  demonstrations, deterministic random-reset evaluation, 100 paired episodes per task. One bridge
  per task. These state-policy cells establish action-rate transfer and are not called VLA cells.
- Visual RoboCasa confirmation: CloseSingleDoor, RGB from the recorded external camera plus robot
  proprioception, the same ARC head and rates, 35 training demonstrations and 15 held-out replay
  initial states. This is an observation-backbone validation, reported separately from the
  state-policy primary analysis because its smaller paired sample does not support the same power.

Rates are `k={1,2,4}`. The trained rate set, eight output blocks, training budgets, tasks, arms,
and sample counts may not change after results.

## Claims and tests

Primary: at `k=2`, ARC+TAC-Fold beats each of (i) step-head+post-hoc TAC-Fold, (ii) raw cubic
spline, and (iii) raw bounded-error B-spline on Push-T and on the pooled non-void manipulation
cells. Exact paired McNemar per comparison, with Holm correction across the six declared primary
comparisons. A cell is void only if both compared arms are below 10% or above 90%.

Mechanism:

1. `arc+tac_fold` must beat `arc+zoh` on at least one non-void `k=2` or `k=4` cell after Holm.
2. ARC at `k=1` must be non-inferior to the step head with a paired-bootstrap 90% lower bound above
   -5 percentage points; otherwise the multi-rate representation buys robustness by damaging native
   control and the claim fails.
3. `k=4` is reported everywhere but may be a floor on Push-T. No method claim is drawn from a void
   cell.
4. The term “visual policy” is allowed only if the visual CloseSingleDoor checkpoint and paired
   results are present. No language-conditioned claim is made: the benchmark supplies no language
   input. “VLA” may describe the architecture family, never the evaluated input modalities.

The defensible claim, if the gates hold, is: “joint rate conditioning removes part of the post-hoc
coarsening penalty, while conservative TAC-Fold decoding preserves the commanded displacement.”
It is not “TAC-Fold universally beats splines,” “task execution is faster,” or “state inputs are a
VLA.”

## Pre-launch clarification and safety amendment — 2026-09-18

This amendment was recorded before any `result_*arcmain*.json` file or ARC campaign checkpoint
existed. It changes no task, arm, rate, training budget, evaluation budget, or outcome.

1. ARC continuous block means are clipped elementwise to the controller command interval
   `[-1,1]` before multiplication by `k`. Therefore every commanded block sum satisfies
   `|S_d| <= k`, which makes bounded exact-sum reconstruction feasible even when an unconstrained
   network prediction falls outside the training range. Saturation incidence is reported.
2. “Beats” in the six primary comparisons means a positive paired success-rate difference with an
   exact two-sided McNemar result that remains below `alpha=0.05` after Holm correction. A void
   comparison cannot pass the primary gate.
3. The native-rate no-harm estimand is the paired success difference pooled over all eight declared
   state-policy tasks. Its 90% paired-bootstrap lower confidence bound must exceed `-5` percentage
   points. Task-level intervals are supporting heterogeneity checks and cannot replace the pooled
   gate.
4. The visual gate requires one shared CloseSingleDoor ARC checkpoint, complete paired results at
   all three declared rates, the fixed 35/15 split and 15k-step budget, and matching checkpoint
   hashes. Its `n=15` comparisons are supporting evidence rather than significance claims.
