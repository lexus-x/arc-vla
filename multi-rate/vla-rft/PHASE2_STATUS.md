# Phase 2 Status: GRPO Flow-SDE RL Pilot on SmolVLA (LIBERO-Spatial)

Read-only on `multi-rate/vla-vault`'s `output/`/`wiki`/ledgers throughout. All new
code/results in `/home/user/Desktop/multi-rate/vla-rft/`.

**Top-line: baseline reproduction is COMPLETE and real. RL pilot training is
VALIDATED end-to-end (a real gradient step occurred) but was still running
in the background (nohup, disowned, survives this report) when this status
was written — it did not reach all 8 planned updates or a final RL-checkpoint
eval within this session. That is reported here plainly, not hidden.**

## Coordinator speed-up request — addressed

A mid-task message asked to exploit GPU/VRAM headroom and shard rollout
collection across the other 2 GPUs (Windows A6000, AWS A6000).

- **VRAM/batch size**: confirmed real headroom (`nvidia-smi`: peak ~19GB RSS
  process memory, GPU **memory** never exceeded ~6GB of 97GB with both the
  baseline eval and the pilot running concurrently; GPU **utilization** capped
  at 27-41%). The actual bottleneck is CPU-bound MuJoCo/robosuite env stepping
  and per-step Python/model-call overhead (56 cores, baseline alone sustained
  350-400% CPU), not GPU compute or VRAM. A batched-multi-env rollout rewrite
  would help, but restructures the sequential single-env rollout loop
  (`train_grpo.py::rollout_episode`) into a vectorized one — judged too risky
  to retrofit into an already-progressing, validated pilot mid-run. Flagged
  below as the concrete next-step lever instead.
- **Cross-GPU sharding**: **blocked, verified, not a judgment call.**
  `~/.ssh/config` does not exist on this Blackwell machine; the SSH aliases
  named in the vault's `CLAUDE.md` (`blackwell`, `a6000`, `2xa6000-islab01`,
  etc.) belong to the laptop orchestrator's config, not this host's. A direct
  connectivity probe to those alias names from here timed out (no resolution).
  This session has local shell access to Blackwell only. Separately: the
  vault's own `CLAUDE.md` lists `aws` as a **RESTRICTED HOST** — the
  coordinator's message said all 3 GPUs were fair game; flagging this
  discrepancy rather than silently picking a side.
- **Tighter pilot check**: done — a real (non-degenerate) gradient update was
  observed at update 1, ~12.5 minutes of wall-clock into the pilot, and
  reported here rather than waiting for the full run.

## Task 1 — 3-seed SFT baseline re-eval on LIBERO-Spatial

**TASK:** Reproduce `flow8300_s0/checkpoints/8300`'s LIBERO-Spatial success
rate ourselves, 3 seeds, to get our own noise-floor stddev (not just trust
the vault's cited 79.40%±1.56%, which is training-seed variance across 5
independently-trained models — a different quantity from eval-time variance
on one fixed checkpoint, which is what an RL comparison actually needs).

**COMMAND:** `conda run -n vla_smolvla_libero python baseline_eval.py`
(source: `/home/user/Desktop/multi-rate/vla-rft/baseline_eval.py`)

**SOURCE:** Reuses the vault's own `evaluate_multirate_honest.py` exactly as
`run_matched8300_spatial.sh` does (same `MODELS`/`ARMS` registration
pattern), looped under 3 global seeds. `env.init_state_id` is pinned per
trial by the harness itself (the "init_state_id drift bias" trap fix already
in that code) — so init states are **identical** across our 3 runs by
design; what varies is the policy's own flow-matching sampling noise
(unseeded `torch.randn` inside `sample_actions`), which is exactly the
right thing to measure since our RL eval will also be stochastic.
Scale: `--trials_per_task 10` (n=100/seed on the 10-task suite), reduced from
the vault's canonical `--trials_per_task 30` (n=300) to fit this session's
time budget — stated explicitly, not hidden.

**RESULT (n=300 total, 100/seed):**

| seed | aggregate | n |
|---|---|---|
| 0 | 72.0% | 100 |
| 1 | 82.0% | 100 |
| 2 | 78.0% | 100 |

**Mean: 77.33%, std: 5.03% (ddof=1, n=3 seeds).**

**CHECK (executed, output shown above, matches the summary file):**
```
cat /home/user/Desktop/multi-rate/vla-rft/baseline_results/summary.json
```
Full per-task breakdowns are in
`baseline_results/seed{0,1,2}/multirate_honest.json`. Task 5 ("pick up the
black bowl on the ramekin...") scored 10% on seed0, consistent with this
vault's own prior finding that this task is unusually hard (see
`wiki/failures/eval-harness-traps.md`'s task-5 diagnostic note) — a
consistency check that this reproduction is measuring the real thing, not an
artifact.

**Noise-floor implication for any RL claim:** per this whole project's
standing rule, a claimed delta must clear roughly 2x this stddev, i.e.
**≳10 percentage points**, to be treated as a result rather than noise, on
this reduced n=100/seed protocol. (The vault's own n=300/seed protocol would
tighten this; not run here due to time.)

## Task 2 — GRPO training loop

**TASK:** Minimal GRPO loop: Flow-SDE stochastic rollout collection, sparse
terminal success/fail reward, group-relative advantage (no critic), clipped
surrogate via the ported `gaussian_logprob`, `grad_clip_norm=10.0` (from the
checkpoint's own `config.json`).

**SOURCE:** `/home/user/Desktop/multi-rate/vla-rft/train_grpo.py` (new file).
Rollout mechanics replicate `SmolVLAPolicy.select_action`'s queue management
(`modeling_smolvla.py:322-347`) but route chunk generation through
`flow_sde.sample_actions_flow_sde` and run under `torch.no_grad()` (the
sampled action's value is all rollout needs; gradient is recomputed at
update time).

**A real correctness bug was found and fixed while writing this** (zero-faking:
reporting it, not hiding it): the original draft stored the FINAL denoised
action (`x_0`, after all 10 Euler steps) as the replay target for computing
`log_prob` at the update step. But `log_prob` is defined at the ONE chosen
stochastic step, not at `x_0` — scoring against the wrong tensor would have
produced a well-formed but meaningless loss (no crash, silently wrong
gradient direction, exactly the "completed run is not a valid run" failure
mode this vault's own `eval-harness-traps.md` warns about repeatedly). Fixed
by having `flow_sde.sample_actions_flow_sde` also return
`chosen_step_sample` (the intermediate sample at the chosen step) and storing
that instead. See `flow_sde.py`'s `replay_target` docstring and the
`chosen_step_sample` field for the fix.

**Design choices and why (not invented silently):**
- Single inner epoch (rollout under current params, immediately update from
  those same rollouts) rather than multi-epoch PPO-style reuse — avoids
  needing an importance-sampling ratio/clip, which is the standard
  simplification for a first correctness pass.
- `LR=1e-6`: 10x below the checkpoint's own SFT backbone LR (1e-5), since a
  single stochastic-step RL update is higher-variance than a dense BC
  gradient; conservative choice, not benchmarked against alternatives here —
  **UNKNOWN** whether this is close to RLinf's own default (could not confirm
  their exact LR from the scratch-copy source read in Phase 1).
- Pilot scope: `TASK_IDS=(3,5)` reused directly from `evaluate_height_screen.py`
  (a pre-existing 2-task diagnostic scope in this vault, not invented here).

**CHECK:** see Task 3's smoke test and pilot run below — this is the code
under test, not independently checkable in isolation.

## Task 3 — RL training run

**TASK:** Start from `flow8300_s0/checkpoints/8300`, run GRPO, check for a
learning signal before committing to a long run (per the agreed pilot-first
protocol), log progress incrementally.

**COMMAND (smoke, group_size=2, n_updates=1 — completed):**
```
conda run -n vla_smolvla_libero python train_grpo.py --group_size 2 --n_updates 1 \
  --out_ckpt rl_checkpoint_smoketest --log_json logs/train_progress_smoketest.json
```
**RESULT:** Ran end-to-end without crashing (env reset, chunked rollout via
Flow-SDE sampler, env stepping, checkpoint save all worked). Both 2-rollout
groups were **degenerate** (task 3: 2/2 success; task 5: 0/2 success — both
zero within-group variance), so `loss=None`, `n_transitions=0`. This is a
real, expected statistical outcome given `group_size=2` and each task's
known extreme success rate (see Task 1's table: task 5 ≈10% success), not a
bug — but it showed group_size=2 is too small to reliably get a gradient
signal, motivating the larger pilot below.

**COMMAND (pilot, group_size=6, n_updates=8 — still running at report time):**
```
conda run -n vla_smolvla_libero python train_grpo.py --group_size 6 --n_updates 8 \
  --out_ckpt rl_checkpoint_pilot --log_json logs/train_progress_pilot.json
```
Launched via `nohup ... &; disown` — **survives this report and keeps running
in the background.** Log: `logs/train_grpo_pilot.log`. Progress (written
after every completed update, real filesystem writes, not buffered like
stdout under `conda run`): `logs/train_progress_pilot.json`.

**RESULT as of this report (4 of 8 updates observed):**

| update | success_rate (12 ep) | n_transitions | loss | grad_norm | elapsed |
|---|---|---|---|---|---|
| 0 | 50.0% | 0 (degenerate) | — | — | 325s |
| **1** | **41.7%** | **667** | **0.810** | **150.48 → clipped to 10.0** | 752s |
| 2 | 50.0% | 0 (degenerate) | — | — | 1074s |
| 3 | 50.0% | 0 (degenerate) | — | — | 1386s |

**Update 1 is the load-bearing result of this pilot: a real, non-degenerate
gradient step happened** — 667 chunk-decision transitions were replayed with
gradient, loss computed, gradient norm (150.48, correctly exceeding the
un-clipped smoke-test's ~52k-per-single-step scale down to a per-batch-mean
150) clipped to the configured 10.0, optimizer step applied. This is the
"the RL loss is computable and a real update occurs" existence proof the
whole pilot exists to establish, and it passed.

**Updates 0 and 2 being degenerate is itself a genuine, useful finding, not
noise to explain away:** with `group_size=6` and task success rates at the
extremes (task 3 usually succeeds, task 5 rarely does, per Task 1's table),
the probability that all 6 rollouts in a group land on the same outcome by
chance is high (e.g. ≈53% for a task at either a 90% or 10% true success
rate: `0.9^6 ≈ 0.53`). GRPO's group-relative advantage is a real gradient
signal only when a group is mixed — this pilot is directly demonstrating
that limitation in practice, on this exact checkpoint and these exact tasks,
which is worth keeping for the eventual paper's limitations/method section
regardless of the final outcome.

**UNKNOWN / not yet done:** whether the full 8-update pilot shows a
success-rate trend (up, down, or flat) — insufficient data (n=3 updates,
2 of them degenerate) to say anything about a trend yet. The process ID and
log paths above are how to check when it finishes.

## Task 4 — Final eval of the RL-trained checkpoint

**NOT DONE.** Blocked on Task 3 finishing (or at minimum, producing enough
non-degenerate updates to be worth evaluating) and on genuine wall-clock
budget for this session. `rl_checkpoint_smoketest/` exists (1 degenerate
update, not worth evaluating — no real update occurred). `rl_checkpoint_pilot/`
will be written when/if the 8-update pilot's `main()` reaches its final
`policy.save_pretrained` call; it does not exist yet as of this report.

## Explicitly unresolved (zero-faking: not claiming these)

- Whether the full pilot shows any success-rate improvement — genuinely
  unknown, not yet enough data.
- `LR=1e-6` and `group_size=6`'s appropriateness vs. RLinf's own
  hyperparameters — not benchmarked, stated as a reasoned but unverified
  choice above.
- Final RL-vs-baseline comparison with the noise-floor check (Task 1's
  ≳10pp bar) — cannot be done until Task 4 exists.
- Whether `flow_sde.py`'s ported mechanism is byte-identical to current
  upstream RLinf (carried over as unresolved from Phase 1 — still not
  diffed against a fresh clone).

## Files produced this phase

- `/home/user/Desktop/multi-rate/vla-rft/baseline_eval.py` (script)
- `/home/user/Desktop/multi-rate/vla-rft/baseline_results/{seed0,seed1,seed2}/multirate_honest.json`, `summary.json` (real data)
- `/home/user/Desktop/multi-rate/vla-rft/train_grpo.py` (script, includes the fix described in Task 2)
- `/home/user/Desktop/multi-rate/vla-rft/flow_sde.py` (extended from Phase 1: `replay_target`, `chosen_step_sample`, no-grad optimization before/skip-after the chosen step)
- `/home/user/Desktop/multi-rate/vla-rft/logs/{baseline_eval.log, train_grpo_smoketest.log, train_grpo_pilot.log, train_progress_smoketest.json, train_progress_pilot.json}`
- `/home/user/Desktop/multi-rate/vla-rft/rl_checkpoint_smoketest/` (exists, not meaningful — see Task 3)

## Recommended next steps (for whoever picks this up)

1. Check `logs/train_progress_pilot.json` for the completed 8-update pilot
   (background job, PID was 1214418 for the worker process at report time,
   may have exited by the time this is read — check via the log/json files,
   not the PID).
2. If any update shows a success-rate trend worth reporting, run Task 4
   (final eval of `rl_checkpoint_pilot/`, same protocol as Task 1) before
   claiming anything.
3. For real speed-up (per the coordinator's request, only partially
   actionable this session): shard rollout collection across **multiple
   concurrent OS processes on Blackwell itself** (not batching within one
   process) — the vault's own `scratch/launch_parallel_eval_blackwell.py`
   8-shard pattern is the precedent; this exploits the 56 idle CPU cores
   (the real bottleneck, confirmed above) without a risky rewrite of the
   sequential single-env rollout loop. Cross-machine (Windows A6000 / AWS
   A6000) sharding remains blocked pending actual SSH/network setup from
   whichever session has it, and pending resolving the AWS-restriction
   discrepancy noted above.
