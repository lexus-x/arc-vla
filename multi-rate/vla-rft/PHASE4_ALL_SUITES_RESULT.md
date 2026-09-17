# Phase 4: All-4-Suites Result (Final)

Supersedes the earlier version of this file, which documented Object/Goal/Long
baseline eval as blocked by an unresolved measurement bug. That blocker did
not reproduce on a clean rerun into a fresh output directory (see "What the
blocker actually was" below) — all 4 suites now have real, on-disk 1-seed
baseline + RL pilot + RL eval results.

## Task: Object/Goal/Long baseline + RL pilot + RL eval, 1 seed each

**TASK:** For LIBERO-Object, LIBERO-Goal, LIBERO-10 (Long): 1-seed baseline
eval (n=100), 8-update GRPO RL pilot, 1-seed RL eval (n=100), reusing the
`flow8300_{suite}_s0` checkpoints, alongside the already-standing LIBERO-Spatial
result (Phase 1-3).

**COMMAND:** `run_suite_pipeline.sh`'s three steps run individually per suite
(`baseline_eval_suite.py` → `train_grpo_suite.py` --group_size 6
--n_groups_per_update 2 --n_updates 8 → `rl_eval_suite.py`), all in
`/home/user/Desktop/multi-rate/vla-rft/`, conda env `vla_smolvla_libero`.

**SOURCE:**
- `baseline_results/libero_{object,goal,10}_seed0_v2/summary.json`
- `baseline_results/rl_{object,goal,long}_seed0_v2/rl_vs_baseline.json`
- Spatial (unchanged): `PHASE3_FINAL_RESULT.md`, `rl_eval_result.json`

**RESULT (n=100 rollouts/arm/suite, 1 seed each):**

| Suite | Baseline | RL pilot (8 GRPO updates) | Delta | z | p (two-prop) |
|---|---|---|---|---|---|
| Spatial | 72.0% (same-seed) / 77.33%±5.03% (3-seed mean) | 80.0% | +8.00pp / +2.67pp | 1.32 (vs same-seed) | 0.1853 |
| Object | 86.0% | 88.0% | +2.00pp | 0.42 | 0.6741 |
| Goal | 93.0% | 91.0% | **−2.00pp** | −0.52 | 0.6022 |
| Long (10) | 55.0% | 58.0% | +3.00pp | 0.43 | 0.6687 |
| **Pooled** | **76.5%** (306/400) | **79.25%** (317/400) | **+2.75pp** | **0.94** | **0.3488** |

**CHECK:** `python3 verify_all_suites.py` — re-derives the Object/Goal/Long
counts directly from `rl_vs_baseline.json` on disk and asserts they match the
table above, then recomputes the z-test/p-value from those counts. Executed,
passes clean.

Vault's own independently-recorded same-day baseline numbers (`output/matched8300_{suite}.json`)
for context, not used in the delta above (different eval invocation/day):
Object 90.0%, Goal 94.0%, Long 62.0%. Our v2 baselines (86.0% / 93.0% / 55.0%)
land in the same neighborhood, which is the actual evidence the earlier 0%/2%/13%
readings were a measurement artifact and not a real baseline number.

## What the blocker actually was

The original Object/Goal/Long baseline runs (`baseline_results/libero_{object,goal,10}_seed0/`)
scored 0.0% / 2.0% / 13.0% — a measurement artifact, root-cause unresolved but
narrowed: `baseline_eval_suite.py` already carried the MODEL_DATASET
normalization-stats fix at the time, and a same-day repro attempt
(`debug_repro_object/`, 20:36) with that fix in place *still* got 0%. Rerunning
the identical script into a fresh `_v2` output directory later got 86% for
Object, 93% for Goal, 55% for Long — same code, same checkpoint, different
result. This points to a session/environment-state issue (stale GPU memory,
a stuck worker process, or similar) rather than a code bug, but this was not
conclusively isolated. **Practical implication: treat any future suite that
returns a suspiciously low/degenerate baseline as "rerun into a fresh output
dir before trusting it," not as a real number.**

## RL training notes (mechanism diagnostics)

Degenerate-update rate (zero within-group reward variance, per PHASE3's
documented GRPO-at-this-scale limitation) varied a lot by suite:
- Object: 0/8 degenerate (`logs/train_progress_object.json`)
- Goal: 0/8 degenerate (`logs/train_progress_goal.json`)
- Long: 5/8 degenerate (`logs/train_progress_long.json`) — sampled rollout
  groups hit 100% success on 4 of 8 updates, leaving no reward variance for
  GRPO to learn from on a suite whose n=100 baseline is only 55%; the training
  subsample (2 groups/update from a task subset) is evidently not
  representative of full-suite difficulty.

## Reference-method comparison table (updated)

| Method | Backbone size | Action head | Benchmark scope | Base (SFT) | Post-RL | Delta | Notes |
|---|---|---|---|---|---|---|---|
| **VLA-RFT** (arXiv:2510.00406) | ~0.5B (VLA-Adapter) | L1 regression (NOT flow-matching) | Standard LIBERO 4-suite | 86.6% avg | 91.1% avg | +4.5pp avg | Small-scale + RL, but not flow-matching |
| **π_RL** (arXiv:2510.25889) | ~3B (π0/π0.5) | Flow-matching (Flow-Noise/Flow-SDE) | LIBERO (numbers not independently reconfirmed — UNKNOWN) | — | — | — | Flow-matching + RL, but large-scale |
| **SimpleVLA-RL** (arXiv:2509.09674) | 7B (OpenVLA-OFT) | Discrete/parallel-decode | LIBERO aggregate | 91% | 99% | +8pp | Neither small-scale nor flow-matching |
| **Ours (this project)** | ~450M (SmolVLA) | Flow-matching | Standard LIBERO 4-suite, 1 seed/arm | 76.5% pooled | 79.25% pooled | **+2.75pp pooled, p=0.35 — not significant** | Only entry combining <500M + flow-matching + RL + standard LIBERO |

**Positioning claim (unchanged, independent of the numeric result):** still the
only entry combining sub-500M scale, a flow-matching action head, RL
post-training, and the standard LIBERO 4-suite protocol.

## Honest verdict — apply the 4-question rubric (`paper-evaluation.md`)

1. **Is it REAL?** — **No.** 1 seed/suite, not ≥3. Pooled p=0.35 across n=400/arm.
   No individual suite clears p<0.05. Goal's delta is negative. This is not
   distinguishable from noise on the evidence collected so far.
2. **Is it CAUSED by the mechanism?** — **Not established.** No ablation with
   the RL mechanism toggled off under matched compute exists; the pilot shows
   the flow-matching GRPO machinery produces real (non-degenerate) gradients
   on 3 of 4 suites, which is a mechanistic existence proof, not an
   attribution result.
3. **Is it NOVEL?** — **Yes, holds.** Positioning claim (sub-500M + flow-matching
   + RL + standard LIBERO) survived adversarial novelty review and is
   unaffected by today's numeric result either way.
4. **Does it MATTER?** — **No, not yet.** +2.75pp pooled, sign-inconsistent
   across suites (3 positive, 1 negative), against real GPU-hour cost per suite
   (~75–95 min RL training + eval). Too small and too inconsistent to claim a
   practical gain on this evidence.

**Verdict: REVISE, bordering on KILL as a publishable numeric claim at this
evidence level.** Three of four rubric questions are weak or fail (REAL,
CAUSED, MATTER); only NOVELTY holds. The mechanism genuinely works
(non-degenerate GRPO updates on flow-matching heads at <500M scale, the open
technical question this project set out to answer) and the pooled point
estimate is still positive, so this is not evidence *against* a real effect —
but it is not evidence *for* one either, and Goal's negative flip means "RL
reliably helps" cannot be claimed even qualitatively.

The only fix that would move this to REAL is the ~15–16 seed/arm campaign
already power-analyzed earlier in this project (previously estimated at
2.5–4 weeks of GPU time) — a scope decision, not a bug fix. Absent that
investment, the honest published claim from current evidence is narrower:
*"RL post-training on a flow-matching action head at <500M scale produces
mechanistically real gradient updates and a small positive pooled effect
(+2.75pp, n=400/arm) that does not reach significance and is not consistent
in sign across LIBERO's 4 suites."* That is a legitimate, if modest, finding
— a mechanism-existence result, not a performance claim.

## Files

- `verify_all_suites.py` — stats CHECK script (stdlib two-proportion z-test)
- `baseline_results/libero_{object,goal,10}_seed0_v2/` — fixed baselines
- `baseline_results/rl_{object,goal,long}_seed0_v2/` — RL eval vs baseline
- `logs/train_progress_{object,goal,long}.json` — per-update GRPO diagnostics
- `rl_checkpoint_{object,goal,long}/` — trained checkpoints
