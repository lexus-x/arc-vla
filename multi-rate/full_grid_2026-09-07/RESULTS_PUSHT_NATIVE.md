# Results — Push-T native reproduction on gym-pusht

Written 2026-09-16 17:19 KST, against `PREREG_PUSHT_NATIVE.md`. Single eval, step 300000
checkpoint of `pusht_native_dp_run`.

## Result

`pusht_native_dp_run/outputs/eval/step_300000_real/eval_info.json`:

| metric | value | n_episodes |
|---|---|---|
| `avg_max_reward` (paper coverage metric) | **0.5295** | 50 |
| `pc_success` (binary, non-comparable diagnostic) | 24.0% | 50 |
| `avg_sum_reward` | 58.03 | 50 |

## Gate — FAILED

`PREREG_PUSHT_NATIVE.md:56-57`, P1: *"reproduced coverage score lands within 5pp of 72%."*

**52.95% vs 72% = 19.05pp short. P1 fails**, well outside the ±5pp gate and outside n=50's
~±14pp sampling noise. Per `PREREG_PUSHT_NATIVE.md:58-60`, next step is root-causing in the
registered order — (a) demo count/quality vs paper, (b) `dp_min` implementation vs lerobot's
reference `DiffusionPolicy`, (c) training budget — not a bigger retrain.

## Supporting evidence: budget is not the bottleneck

Coverage has been flat since step 40k; the last 260k training steps bought ~0pp:

| step | `avg_max_reward` | `pc_success` | source |
|---|---|---|---|
| 20000 | 0.4285 | 14.0% | `pusht_native_dp_run.log` |
| 40000 | 0.5229 | 26.0% | `pusht_native_dp_run.log` |
| 60000 | 0.5095 | 26.0% | `pusht_native_dp_run.log` |
| 80000 | 0.5369 | 24.0% | `pusht_native_dp_run.log` |
| 100000 | 0.4994 | 14.0% | `pusht_native_dp_run.log` |
| 300000 | **0.5295** | **24.0%** | `outputs/eval/step_300000_real/eval_info.json` |

This points at (a) or (b), not (c) — do not spend compute on a longer/bigger retrain before
checking demo count/quality and the DP implementation against lerobot's reference.

## Prereg deviations (procedural, do not affect the gate verdict)

- `PREREG_PUSHT_NATIVE.md:52` requires eval seeds recorded in the result JSON. Not present —
  `eval_info.json` only carries `sum_rewards, max_rewards, successes, video_paths`.
- `PREREG_PUSHT_NATIVE.md:27` specifies state-only 5-dim obs. This run used `pixels_agent_pos`
  at 384×384 (matching the checkpoint's `train_config.json`, not the prereg's obs_type).

## Not allowed (per prereg, restated)

Do not place this row in the same table column as ManiSkill `PushT-v1` binary-success numbers
(`PREREG_1X.md:14`) — different env, different metric. Do not relabel `pc_success` as the
72%-comparable number.
