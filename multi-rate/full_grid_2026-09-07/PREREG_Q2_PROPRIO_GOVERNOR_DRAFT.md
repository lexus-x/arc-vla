# Pre-registration DRAFT — proprio-only learned governor vs post-hoc converters (Q2 confirmatory)

**Status: DRAFT, not sealed.** Written 2026-09-28 KST. No episode of this campaign has been evaluated.
Sealing = user approval + sha256 of every frozen file appended below, before the first eval episode.

## Why this campaign exists
- Sealed campaign `PREREG_LEARNED_GOVERNOR.md` (commits 2f02587, 988ee72): H1 PASS at k=4 (5/8) and k=2 (4/8),
  but H3 fired: a 2x256 MLP on full state (`mlp_bc`, no DP) ≥ the governor on 6/8 tasks at k=4.
  With full object state the governor is a second policy, so "the small net explains the gain" cannot be excluded.
- Fix chosen by the user (2026-09-26), a priori: the converter and every learned ablation see **only proprioception +
  the frozen policy's coarse commands**. The gain must then come from interpreting the policy, not from re-solving the task.
- Also added: 3 training seeds, fresh initial states, B-Spline Policy's own benchmarks, official BSP as the retrained reference.

## Dev evidence already seen (training demos only, NOT citable)
- `dev_proprio_offline.json`: proprio-only governor held-out-demo MSE ≈ full-state governor on all 16 task×k cells
  (within ±10%), 3–70x below ZOH. No closed-loop number of the proprio governor exists.
- `bsp_official.py` conformance: official targets → official plan/align loop reproduce training demos (fit p95 ≤ 0.0084
  on all 8 ManiSkill tasks, 8.3–9.6 executed steps/plan vs DP's 8).

## Method (frozen at sealing)
`shape_governor.py` recipe unchanged (MLP 2x256 GELU, AdamW 1e-3, wd 1e-4, 4000 steps, batch 256, 10% demo holdout),
input = 2-frame **proprio slice** + block means of the executed 8-step chunk; anchor → `resample_qp_anchor`.
Governor fit seed = the DP training seed it is paired with.

**Proprio slice (mechanical rule: robot-internal state only):**

| Benchmark | Proprio = | dims |
|---|---|---|
| ManiSkill Panda tasks (7) | `agent.qpos`, `agent.qvel` (leading dims of the state vector) | 18 |
| AnymalC-Reach | `agent.qpos`, `agent.qvel` | 24 |
| RoboMimic lift/can/square | `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` (last 9 of `RM_OBS`) | 9 |
| RoboCasa | every low-dim key with prefix `robot0_` | per task |
| Push-T (gym-pusht) | `agent_pos` | 2 |

The frozen policy (DP / BSP) keeps its full observation. Only converter-side learned components are restricted.

## Arms (paired on identical initial states and diffusion noise)
- Method: `qp_learned_p`.
- Post-hoc converters (the comparison family): `zoh`, `spline_satfix`, `tac_fold_satfix`, `qp_anchor`.
- Ablations, same proprio inputs: `learned_raw_p` (no QP), `learned_tanh_p` (bounded net, no QP), `mlp_bc_p` (small net as the policy, no DP).
- References: `native` (frozen DP at its trained rate, ceiling) and `bsp` = official B-Spline Policy, **retrained** with the
  same backbone/demos/steps, executed rate-free at native speed with the official time-alignment (`bsp_official.plan`).
  BSP is paired by initial state only (its own policy and noise).

## Policies
- DP step head, seeds 0/1/2, `--checkpoint-suffix _q2`, 30k steps, trained 2026-09-28 with `--n_eval 0` (no episode touched).
  The sealed-campaign checkpoints `dp_<task>.pt` are **not** reused.
- BSP head (`--head bsp`), seeds 0/1/2, same demos and budget.

## Benchmarks, windows, n
| Family | Tasks | demos / steps | Fresh initial states | n per seed |
|---|---|---|---|---|
| ManiSkill | PickCube, RollBall, PullCube, LiftPegUpright, PushCube, AnymalC-Reach, PokeCube, StackCube | 200 / 30k | env reset seeds 200000+i (all demo seeds ≥ 1717782; asserted disjoint at runtime) | 300 |
| RoboMimic | lift, can, square | 200 / 30k | reset seeds 10000+(5000+i) (prior evals used 10000+0..399) | 300 |
| RoboCasa | TurnOffSinkFaucet, CoffeePressButton, TurnOffMicrowave, CloseSingleDoor (BSP Table 2a) | **OPEN** / 15k | `--rc-random-eval`, new `--eval-seed` block | 100 |
| Push-T | gym-pusht native (BSP's env) | **OPEN** | new seed block | 300 |

Held-out demo states are exhausted (PickCube 0 left; others 130–322), hence fresh resets for ManiSkill.
Rates: **k=4 primary**, k=2 secondary (run after all k=4 cells).

## Hypotheses and tests
Exact two-sided McNemar on episode pairs pooled over the 3 seeds. VOID if both arms <10% or both >90% (pooled).
- **H1 (primary, k=4):** `qp_learned_p` beats the post-hoc converters. Per task, intersection-union test: win requires
  `qp_learned_p` > each of `spline_satfix`, `tac_fold_satfix`, `qp_anchor` (task p = max of the three p-values),
  **and** a positive point estimate vs each of them in every seed separately. Holm across all tasks run.
  **Claim rule:** Holm wins on ≥ half of the non-void tasks, ≥1 win outside ManiSkill, 0 Holm losses on any task.
- **S1 (k=2):** H1 rules, reported separately.
- **S2 (fixed sequence, only if H1 passes):** non-inferiority vs `bsp` at each k: one-sided 97.5% lower bound (Tango score
  interval, paired) of `qp_learned_p − bsp` > −5 pp. Reported per task; no superiority claim over BSP is pre-registered.
- **S3 (mechanism, directional, Holm):** `qp_learned_p` > `learned_tanh_p` (the k=4 H6 null vs the k=2 exploratory wins).
- **S4 (no direction, Holm):** `qp_learned_p` vs `mlp_bc_p`. If `mlp_bc_p` ≥ `qp_learned_p` on most tasks, the paper says so.
- **S5:** `qp_learned_p` vs `learned_raw_p`, and vs `native` (no-harm context, not a claim).

Power (rough): 900 pairs per task, 15% discordance → ≈5 pp detectable at 80% power after Holm over 16 tasks.

## Not allowed after results
No change to tasks, seeds, windows, arms, proprio rule, recipe, k values, margins or claim rules. No dropping a family,
seed or k. Losses, voids and failed seeds reported as measured. The full-state results of the sealed campaign stay in the paper.

## Open items to settle before sealing
1. Push-T: governor + BSP integration into the native gym-pusht runner (separate from `harness.py`) — include, or drop the
   family now (before any result)?
2. RoboCasa demo count per task (39 in the Sept campaign) and whether the bridge throughput allows n=100 × 3 seeds × 11 arms.
3. NI margin δ = 5 pp — confirm.
4. Code added 2026-09-28 (frozen at sealing): `--gov-obs proprio` (model stores its `obs_idx`), `--head bsp` (bsp_official.plan in
   run_episode), `--ms-random-eval` (fresh resets, runtime disjointness assert), RoboMimic `--eval-offset`. Smoke-tested on a
   throwaway seed block (900000+). DP step checkpoints: `q2_checkpoints.sha256` (24, all 30k steps, n=0 side results).
