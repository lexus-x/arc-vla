# Pre-registration — Push-T native reproduction on gym-pusht (paper's own env)

Written 2026-09-14 20:50 KST, before any training run. Goal: reproduce B-spline Policy
(arXiv 2607.09648) Table 2 PushT base cell (72%) on the environment it was actually measured
on, replacing the ManiSkill `PushT-v1` binary-success number (24%, different env + metric,
see `PREREG_1X.md:14`).

## Protocol, confirmed from source (not memory)

All facts below read directly from the installed packages, not assumed:
- `gym_pusht/envs/pusht.py` (installed 2026-09-14, `gym-pusht==0.1.6`, `pymunk==7.3.0`)
- `/home/user/lerobot/src/lerobot/envs/configs.py::PushtEnv`
- `/home/user/lerobot/src/lerobot/scripts/lerobot_eval.py`
- `/home/user/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py`

| Item | Value | Source |
|---|---|---|
| control_hz | 10 | `pusht.py:165` (`self.control_hz = self.metadata["render_fps"]`, `render_fps=10` at `pusht.py:135`) |
| physics dt | 0.01 (100Hz substeps, 10 substeps/control step) | `pusht.py:166,242` |
| success_threshold | **0.95** coverage (not 0.90 — that was ManiSkill's own choice, unrelated) | `pusht.py:181` |
| reward / coverage score | `clip(coverage / 0.95, 0, 1)` where `coverage = intersection_area(block, goal) / goal_area` | `pusht.py:232-238,257` |
| episode termination | immediate, on first step where `coverage > 0.95` | `pusht.py:258` |
| episode_length (lerobot default) | 300 steps @ 10Hz = 30s | `configs.py:136` |
| **paper coverage metric** = lerobot's `avg_max_reward` | mean over eval episodes of `max_t clip(coverage_t/0.95, 0, 1)` | `lerobot_eval.py:364-365,445` |
| binary success (secondary, report alongside) | `pc_success` = fraction with any `is_success=True` step | `lerobot_eval.py` (`successes`/`pc_success`) |
| DP hyperparams (reuse, matches our `dp_min.py` already) | n_obs_steps=2, horizon=16, n_action_steps=8 | `configuration_diffusion.py:104-106` |
| obs_type | `state` (5-dim: agent_x,y, block_x,y, block_angle) — matches our state-only pipeline, skip pixels | `pusht.py:184-190` |

Session note claim of "200Hz" (`.remember/today-2026-09-11.md:8`) does not match source —
control_hz is 10, physics substep is 100Hz. Treat the 200Hz note as wrong; source wins.

## Data

`lerobot/pusht` (HuggingFace hub) — human teleop demos, the dataset the paper and lerobot's DP
baseline both train on. Not yet downloaded locally (checked HF cache — absent). Pull via
`LeRobotDataset` in step 3. Use whatever episode count ships in that dataset (report the actual
count in the results table — do not assume 90 or 206 without checking).

## Training budget

Reuse `dp_min.py`'s `DiffusionPolicy` (same implementation used in every other table cell —
do not swap implementations mid-comparison). Train to convergence on loss curve, same stopping
discipline as `train_pusht_matched.py` (537k steps got loss to 1e-4 on our current
demo/env combo — use that as a sanity ceiling, not a target; this is a different dataset/env so
convergence step count may differ).

## Eval

- n_eval episodes: 50 minimum (matches our existing table cells' convention)
- report BOTH `avg_max_reward` (== paper coverage metric, compare to 72%) and `pc_success`
  (binary, our own diagnostic — not paper-comparable)
- fixed eval seeds, recorded in the result JSON

## Predictions / gate

- P1 (primary): reproduced coverage score lands within 5pp of 72%. If so, env/metric
  reproduction is validated → proceed to k-rate resampler cells on gym-pusht.
- P2: if coverage is well under 72%, root-cause in this order before retraining bigger:
  (a) demo count/quality vs paper, (b) `dp_min` implementation vs lerobot's reference
  `DiffusionPolicy`, (c) training budget.

## Not allowed

Choosing a different coverage definition after seeing results. Relabeling `pc_success` as the
72%-comparable number. Mixing ManiSkill PushT-v1 rows and gym-pusht rows in the same table
column without a header distinguishing them.
