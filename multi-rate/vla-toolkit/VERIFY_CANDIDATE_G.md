# VERIFY — Candidate G, progress-shaped GRPO reward

n=40 episodes/checkpoint (TASK_IDS=(3,5), 20/task), 2 seeds, matched compute
(n_updates=15, identical script/hyperparameters, only `--lambda_progress`
differs: 0.0=vanilla, 0.5=progress-shaped). Binary success only (shaped reward
used for training, not for this eval).

## Results (pgr_stats.py, 2026-09-04)

| | vanilla | pgr | delta | p (z-test) | paired (McNemar) |
|---|---|---|---|---|---|
| seed0 | 25/40 = 62.5% | 22/40 = 55.0% | -7.5pp | 0.496 | vanilla wins 3, pgr wins 0, p=0.248 |
| seed1 | 21/40 = 52.5% | 20/40 = 50.0% | -2.5pp | 0.823 | vanilla wins 2, pgr wins 1, p=1.000 |
| pooled | 46/80 = 57.5% | 42/80 = 52.5% | -5.0pp | 0.525 | — |

## Honest reading
No significant effect. Both seeds trend negative — progress-shaped reward
underperforms vanilla GRPO, doesn't beat it. Paired comparison agrees:
vanilla wins the shared-seed matchup more often in both seeds, never the
reverse by more than 1.

**Pattern across the whole session, not just this candidate:** four
independently-designed interventions built on the same progress signal (r≈0.99
estimator) all failed to show a positive effect — PGAD noise-lowering
(VERIFY.md), PGAD noise-raising pilot, and now progress-shaped GRPO. A fourth
(replan-cadence-gating) and fifth (denoising-step-gating) died to prior art
before reaching evaluation. This is a consistent, convergent null-to-negative
result, not a one-off unlucky draw.

## Rubric
- **NOVEL**: holds (6-10 target) — learned dense-progress reward shaping,
  differentiated from Prism-GRPO's sim-privileged contact heuristic. Unaffected.
- **REAL**: FAILS. No direction clears significance; trend is negative in both seeds.
- **CAUSED**: moot — nothing to attribute a cause to when there's no effect.
- **MATTERS**: FAILS as a performance claim.

## Verdict: KILL
Not REVISE — the 2-day deadline is now spent, and five separate
mechanisms (2 evidence-tested empirically, 3 killed by prior art or evidence)
have not produced a positive result on this axis. Continuing to iterate is not
viable within the stated budget.

## What's left behind, reusable
- 4 per-suite progress-regression heads (r≈0.99), trained and saved
  (`progress_head_*.pt`) — the signal itself is real and accurate; every
  mechanism built ON TOP of it to change VLA behavior has failed so far.
- A working seed-paired eval harness, stats pipeline (two-proportion z-test +
  paired McNemar), and a GRPO training fork with reward-shaping wired in
  (`train_grpo_pgr.py`) — reusable if a future session wants to try a
  different shaping function or scope.
