# Training-free resampler comparison

Fixed before launching this campaign; historical results have already been inspected.
This is an exploratory benchmark extension, not an independent confirmation of a
winner selected from those historical results.

- Eight tasks: PickCube-v1, PullCube-v1, PushCube-v1, PokeCube-v1,
  LiftPegUpright-v1, StackCube-v1, RollBall-v1, AnymalC-Reach-v1.
- k=2 and k=4. Existing seed-0 step-head DP checkpoints only; no training.
- Same held-out windows as each result_dp_TASK_kK_confirm.json: n=400,
  offset=100 except PickCube n=293, offset=500. Fixed 200 training demos.
- Fresh paired GPU-batched live rollouts for every arm, separate from scalar
  outcomes. Reuse only complete results from this exact new campaign on restart.
- Arms: native, zoh, tac_fold, spline, bspline_eps_raw,
  tac_fold_satfix, spline_satfix, bspline_eps_satfix, qp, qp_anchor.
  SATFix is applied equally in the repaired-method comparison. The existing
  adaptive B-spline implementation uses fixed eps=.005; training-free does not
  mean free of numerical settings. Exclude the known-broken legacy `bspline`.
- Original evaluator's FP32 policy, DDIM-10, eight-action execution chunks,
  controller, horizon, per-episode noise seeds, gripper hold and success criterion.
  No shortening horizons, dropping failures or tuning methods after results.
- Primary descriptive ranking: equal-weight mean success across all 16 task/rate
  cells, separately for raw and SATFix methods. Also report every cell and rate.
  No overall ranking until all 16 cells complete. This suite is saturation-screened
  ManiSkill, so its winner is not automatically best on other suites or policies.
- Pairwise exact McNemar within each cell. Holm across all 48 pairwise contrasts
  among the three SATFix methods; separate family of 48 for the three raw methods.
  No pooled episode test across repeated tasks/rates. QP and QP-anchor descriptive.
- Existing PullCube timing result is not merged: this campaign includes additional
  SATFix arms and records all arms together. Preserve checkpoint and normalizer
  hashes, reference files, source hashes, all outcomes and timings.
- Existing training jobs continue untouched. Run one batched evaluation at a time.
