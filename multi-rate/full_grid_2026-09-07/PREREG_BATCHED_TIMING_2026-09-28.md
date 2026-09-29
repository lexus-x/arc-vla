# Fixed-checkpoint batched evaluation / timing validation

Written before this runner produces any success-rate result. This is an execution
validation and exploratory comparison, not a new confirmatory method claim.

- First full-task timing check: PullCube-v1, k=4, 400 episodes, the existing
  confirmation window (200 training demos, evaluation offset 100).
- Arms: native, zoh, qp, tac_fold, tac_fold_satfix, qp_anchor, spline,
  bspline_eps_raw. All arms get fresh rollouts in the identical 400-environment
  layout. Never merge these outcomes with single-environment result files.
- Preserve checkpoint hash from the existing confirmation file, cached training
  normalizers, FP32 policy computation, DDIM-10, 8 executed actions per chunk,
  original controller and horizon, resampler implementations and success criterion.
- Noise for episode index i and replan r is generated with seed 1000003*i+r,
  with the same (1,16,action_dim) CUDA random draw as the original harness.
- GPU batching has measured numerical differences from scalar physics and policy
  inference. The new result is explicitly a separate batched evaluation, not
  evidence of bitwise scalar equivalence. All arms use the same layout and seeds.
- Record every episode, including failures; no adaptive episode-count reduction,
  horizon truncation, method elimination, or replacement of live rollout by replay.
- Report all arms and timing, regardless of ranking. Pairwise exact McNemar
  comparisons in this timing check are exploratory; do not claim significance.
- The small preceding throughput probes used development episodes and are not SR
  estimates. Passing a timing check does not validate every other simulator.
- No training jobs are stopped. Full-suite completion within 30 minutes remains
  a target, not a demonstrated runtime, until measured end to end.
