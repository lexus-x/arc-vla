# Headroom screen: frozen rule (written 2026-09-28 16:05 KST, before any screen episode)

Purpose: choose which benchmark families enter the Q2 prereg (`PREREG_Q2_PROPRIO_GOVERNOR_DRAFT.md`).
This is a dev screen, not citable evidence. It selects on headroom only and never on our method's results.

## Arms and n
- Arms: `native`, `zoh` only. No converter or governor arm is run in the screen.
- n = 50 episodes per arm, policy seed 0 (or the single official checkpoint).
- k ∈ {2, 4, 8}, subject to the chunk rule below.

## Dev seed blocks (never reused for confirmation)
| Family | Dev block |
|---|---|
| Push-T (gym-pusht, lerobot/diffusion_pusht) | env seeds 5000–5049 (prior evals used 1000–1099) |
| Meta-World | env seeds 5000–5049 per task |
| ALOHA transfer-cube | env seeds 5000–5049 |
| RoboMimic (official DP) | env seeds 5000–5049 |
| Block Pushing (official DP-T) | env seeds 5000–5049 |

Confirmatory blocks are chosen in the prereg and must be disjoint from these (asserted at runtime).

## Pass rule
A family passes at the **smallest k** for which all of these hold:
1. The executed action chunk is ≥ 2k steps (so there are at least 2 coarse blocks).
2. On ≥ half of the family's tasks: native ≥ 30% and native − ZOH ≥ 10 pp.

Families with no passing k are reported in the paper as the "no-headroom regime". Their screen numbers are published as they stand.

## Offline dev check (reported, not a gate)
Proprio-only governor held-out-demo MSE vs ZOH, as in `dev_proprio_offline.json`.

## Amendment 1 (2026-09-28 17:25 KST, before any LIBERO screen episode)
The user ruled out policy training on time grounds, so the DP3 Adroit/DexArt fallback is dropped.
Added family: **LIBERO**, official `lerobot/smolvla_libero` (chunk 50, n_action_steps 50, delta EEF + gripper; gripper causal-held).
- Tasks = the 4 standard suites (libero_spatial, libero_object, libero_goal, libero_10). n = 50 per suite per arm = 5 episodes x 10 tasks.
- Dev seeds 5000+ (lerobot eval seeding). The same pass rule applies, with suites as tasks (pass needs >= 2 of 4 suites).
Screen results already in at this point (not changed by this amendment): Push-T (passes k=2), ALOHA (fails), Meta-World (0/9 at k=8).

## Amendment 2 (2026-09-28, before any RoboTwin screen episode)
LIBERO passed at k=8 (4/4 families). RoboTwin 2.0 is screened as a 5th family, as insurance for RoboMimic's marginal pass.
- Checkpoint: official `lerobot/smolvla_robotwin` (chunk 50, n_action_steps 50, 14-d absolute joint targets). Config `demo_clean`, aloha-agilex, cameras head/left/right → camera1/2/3 (the lerobot CI rename map). Env `robotwin`, lerobot 0.6.2 wrapper, `--eval.batch_size=1 --eval.use_async_envs=false`.
- Tasks: the 10 `ROBOTWIN_TASKS` of lerobot's `.github/workflows/benchmark_tests.yml` (a third-party list, not chosen by us): beat_block_hammer, click_bell, handover_block, stack_blocks_two, click_alarmclock, open_microwave, adjust_bottle, lift_pot, stamp_seal, turn_switch. Pass needs ≥ 5 of 10 tasks.
- n = 50 per task per arm, dev env seeds 5000–5049. A seed whose scene raises `UnStableError` at setup is replaced by s + 100000·j (the first stable one). This is fixed by (task, seed) before any action, so it is identical across arms (the official eval skips such seeds too). Seeds 9900+ were burned on the install smoke.
- Episode length = the task's official limit in `task_config/_eval_step_limit.yml`. No scripted-expert seed filter, so the numbers are not leaderboard-comparable. Arms are paired, so the screen is unaffected.
- Rendering: RoboTwin's ray-tracing shader with the **OptiX** denoiser in place of OIDN. sapien's bundled OIDN 2.0.1 has no Blackwell (sm_120) kernels and leaves frames grainy (high-frequency noise proxy 4.06 vs 0.62 with OptiX on one head-camera frame). It applies to every arm (`eval_lerobot_rate.py`).
- ZOH arm: absolute-target displacement path, with gripper dims 6 and 13 causal-held (same as ALOHA; `RATE_ABS=1 RATE_HOLD_IDX=6,13`).
- Staged stop (cannot change a verdict under the pass rule): (a) native on all 10 tasks; (b) ZOH only on tasks with native ≥ 30%, and the family fails at once if fewer than 5 tasks qualify; (c) k = 2, then 4, then 8, stopping at the first k that passes.
- Caveat, stated before any data: RoboTwin's `take_action` TOPP-plans every joint target and executes it to completion (`envs/_base_task.py:1479`). ZOH therefore becomes a straight move to each block-end target plus idle holds; there is no fixed control rate to fall behind. Small headroom is expected. The numbers are published as they stand.

## Amendment 3 (2026-09-28 ~21:15 KST, before any converter episode on these checkpoints/seeds)
User goal restated: significant method wins in ≥ 5 sim families. Q2 H1 wins must beat **each** of `spline_satfix`, `tac_fold_satfix` and `qp_anchor` (PREREG_Q2_PROPRIO_GOVERNOR_DRAFT.md, H1). Native − ZOH headroom therefore does not show a win is possible, so a **converter check** is added to the screen. It runs only these three baseline arms, never the governor.
- Families and k: every family and k where the ZOH criterion held: Push-T k=2 and k=4, RoboMimic k=4, LIBERO k=8. The same checkpoints, tasks, n = 50, dev seeds 5000–5049 and runners as the ZOH screen (`eval_pusht_hub_resamplers.py`, `eval_dp_official_rate.py`, `eval_lerobot_rate.py`). Native and ZOH arms are reused from the screen.
- A task is **open** at k if native ≥ 30% and native − max(spline_satfix, tac_fold_satfix, qp_anchor) ≥ 10 pp.
- A family stays a Q2 candidate if ≥ half its tasks are open at one checked k. Otherwise it is reported as "converters close the gap" and gets no governor run.
- RoboTwin (amendment 2) and any later family get the same check after passing ZOH, at their passing k.

## Amendment 4 (2026-09-28 ~21:25 KST, before any Franka Kitchen episode)
Added family: **Franka Kitchen** (relay-policy-learning / D4RL kitchen, as in the DP paper). Official DP-C checkpoint `low_dim/kitchen/diffusion_policy_cnn/train_0/checkpoints/epoch=1700-test_mean_score=0.580.ckpt` (the highest-scoring kitchen DP checkpoint on the server; the transformer one is 0.574), from the DP authors' server, sha256 recorded at download.
- Runner: the checkpoint's own `KitchenLowdimRunner` via `eval_dp_official_rate.py`, CPU, `dp_official` env. Delta joint-velocity actions in [-1, 1]: all 9 dims are resampled, none held.
- One task (the multitask kitchen env), so a pass needs 1/1. Success = ≥ 4 subtasks completed (the DP paper's p4), taken per episode from the runner's own completed-task count and cross-checked against its `test/p_4`.
- n = 50 per arm, dev seeds 5000–5049, policy noise seed 0. n_action_steps = 8, so k ∈ {2, 4} (k = 8 violates the chunk rule).
- Arms: native, zoh at k = 2 and 4 (ZOH criterion), plus the amendment 3 converter check at the smallest passing k. For wall time the converter arms run at both k at once; only the passing k is read. Seeds 900000+ are burned on the smoke.

### Amendment 4a (2026-09-28 ~21:30 KST, before any Kitchen episode; the first smoke stopped at an assertion before rolling out)
The kitchen checkpoint is `abs_action: true` (Robot_PosAct absolute joint-position targets), not joint velocities. As for ALOHA and the abs RoboMimic checkpoints: the 7 arm-joint dims go through the displacement path anchored at the measured joint positions (obs[0:7] of the newest frame), and the 2 finger dims are causal-held. This replaces amendment 4's "all 9 dims are resampled, none held". Everything else in amendment 4 is unchanged.

## Amendment 5 (2026-09-29 ~01:10 KST, before any VLABench episode)
Added family: **VLABench** (OpenMOSS; MuJoCo 3.2.2 / dm_control 1.0.22, Franka Panda, absolute end-effector targets converted to joint qpos by IK). Checkpoint: official `lerobot/smolvla_vlabench` (chunk_size 50, n_action_steps 50 — the same long-chunk regime LIBERO passed ZOH in at k=8). No per-task success has ever been published for this checkpoint, so this screen is also its first-ever native measurement, not just a headroom check.
- Runner: `eval_lerobot_rate.py --env.type=vlabench` via the `vlabench` conda env (cloned from `robotwin`, which already carries the lerobot 0.6.2 checkout with `env.type=vlabench` support). `MUJOCO_GL=egl`, `--rename_map` per the lerobot vlabench docs (image/second_image/wrist_image → camera1/2/3), `--policy.device=cuda`.
- Action space: 7-D absolute EEF target (xyz + Euler xyz in radians + gripper). **Must use `RATE_ABS=1 RATE_HOLD_IDX=6 RATE_ANGLE_IDX=3,4,5`** — the delta path clips to [-1, 1], which would destroy radian Euler targets; the Euler dims need the `RATE_ANGLE_IDX` unwrap fix added to `eval_lerobot_rate.py` today (dims 3–5 cross the ±π seam, which every arm — including QP-anchor's hard [-1,1] box — would otherwise mishandle as a fake ~2π jump). Self-check for the fix passes (`eval_lerobot_rate.py::self_check`).
- Tasks: the 10 primitive-track tasks (verified against the repo's own task files): select_fruit, select_toy, select_book, select_painting, select_drink, select_chemistry_tube, select_poker, add_condiment, insert_flower, select_mahjong. n = 50 per arm per task, dev seeds 5000–5049. Episode length per the lerobot env default (500) unless VLABench's own `Evaluator` budget (200) is confirmed to be the comparable one — decide before the native pilot, not after seeing its result.
- Pilot first (burned seeds 9900+, small n): confirm the checkpoint actually runs and produces a plausible success rate before committing GPU time to n=50 × 10 tasks. This is not a protocol change — no episode of the pre-registered screen (dev seeds 5000+) has run yet.
- Same staged rule as every other family: native on all 10 tasks → ZOH only on tasks with native ≥ 30% (fail at once if fewer than 5 qualify) → k = 2, then 4, then 8, stopping at the first k that passes. Then the amendment 3 converter check at the passing k.

**Pilot result (2026-09-29 01:42 KST, burned seeds 9900+): VLABench is not a Q2 candidate.** `lerobot/smolvla_vlabench` scored 0/14 across 5 diverse tasks (select_fruit 0/2, add_condiment 0/3, select_book 0/3, select_drink 0/3, insert_flower 0/3) — 0% native success everywhere tried, far below the 30% floor. Ruled out as a harness bug first: the raw VLABench env (no lerobot wrapper) loads/resets/steps cleanly on all 10 tasks (`vlabench_smoke.py`); the language-instruction path is correctly wired (`lerobot_eval.py` reads `env.call("task_description")` into `observation["task"]`, and `vlabench.py:251-257` populates it from the task object); the `--rename_map` convention is the same one already validated by LIBERO's passing runs. The checkpoint's own train config shows only ~20k steps / batch 32 (~640k samples) — evidently undertrained for this benchmark's semantic pick-and-place tasks. No full n=50 screen was run (would have wasted 3–8 GPU-hours on an already-failed family). The `RATE_ANGLE_IDX` unwrap fix added to `eval_lerobot_rate.py` for this attempt stays in the codebase (harmless no-op for every other family, since none of them set that env var) in case a future VLABench checkpoint is worth retrying.
