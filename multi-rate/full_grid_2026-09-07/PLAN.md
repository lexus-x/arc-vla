# Plan: decimated-arm comparison for the vision policy, CloseSingleDoor only

## Context

Native (k=1) sanity check confirmed `CloseSingleDoor`'s vision-based checkpoint
is credible against the reference paper (ours 40.0%, paper 27% — we exceed
it). This is the one task, of the four Stage 1 tasks, where the baseline is
trustworthy enough to run a real method comparison on. The other 3 remain far
below the paper and are out of scope for this task.

`stage1_vision.py` currently only implements `native_rollout` (k=1, no
decimation, by original design — see its own docstring). This plan adds
decimated-arm rollout, reusing the already-validated resampler math in
`resample_math.py` rather than reimplementing anything.

## What to build

1. In `stage1_vision.py`, add a decimated rollout path parallel to
   `native_rollout`, mirroring exactly how `harness.py`'s main loop calls
   `apply_arm(chunk, arm, n_hold)` on each predicted 8-step chunk before
   executing it (see `harness.py` around its eval loop, and `apply_arm` /
   `decimate_and_resample` / `coarsen_gripper_transitions` / `gripper_sync` in
   `resample_math.py` — import and reuse these directly, do not duplicate the
   math). The vision policy's predicted chunk shape/clip/execution semantics
   are otherwise identical to the state-only policy's (8-step chunks, same
   clip to [-1,1]); only the observation encoding differs, which
   `native_rollout` already handles correctly — copy that part unchanged.
2. `n_hold=6`, `max_steps=500` for `CloseSingleDoor` (confirmed in
   `harness.py`'s `ROBOCASA` dict — use these exact values, don't hardcode
   different ones).
3. Add a `--k` argument (default 2, matching harness.py's convention) and an
   `--arms` argument (comma list) to the eval CLI. Support at minimum:
   `native, zoh, spline_satfix, bspline_eps_satfix, gripper_sync`.
4. Paired execution: reuse the same seeding discipline `harness.py` already
   uses (`torch.manual_seed(1_000_003 * ei + replan)` re-seeded per
   (episode, replan) — grep `harness.py` for the exact line) so all arms see
   identical policy proposals whenever their observations agree, making exact
   McNemar valid.
5. Write a structured result JSON per run (same shape/spirit as
   `harness.py`'s output: per-arm success list, per-episode step counts) so
   the existing `exact_mcnemar` helper in `harness.py` can be reused directly
   to compute paired contrasts — don't write a second implementation of
   McNemar.
6. Add focused tests (small dims, no RoboCasa/network/CUDA required) for: the
   decimate-then-execute wiring produces the same 8-step-chunk semantics as
   `harness.py`'s `apply_arm` given the same input by construction (import and
   call the same functions — a wiring test, not a reimplementation test), and
   that `--arms` selection/CLI parsing works.

## Run (I will execute this myself; sandbox blocks bridge sockets for you)

k=4, n=15 first (matches what we already have for native), all 5 arms:
`native, zoh, spline_satfix, bspline_eps_satfix, gripper_sync`, task
`CloseSingleDoor` only. Reuse the existing checkpoint
(`stage1_vision_results/CloseSingleDoor.pt`) — do not retrain.

## Do NOT

- Do not touch `TurnOffSinkFaucet`, `CoffeePressButton`, `TurnOffMicrowave` —
  their baselines are not credible yet (Stage 2 pending), building a decimated
  comparison for them now would produce meaningless numbers.
- Do not retrain any checkpoint.
- Do not modify `harness.py`, `resample_math.py`'s existing functions, or the
  state-only bridge/eval path.
- Do not claim significance at n=15 — report exact McNemar p-values
  unstarred, same convention as everywhere else in this repo.
