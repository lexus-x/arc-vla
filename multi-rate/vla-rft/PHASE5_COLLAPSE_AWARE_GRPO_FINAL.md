# Phase 5: Collapse-Aware GRPO Campaign — Final Result

Supersedes PHASE4 as the current state of the RL-post-training chapter. 16/16
runs complete: 4 LIBERO suites x 2 conditions (vanilla GRPO vs. "fixed" —
full task coverage + dynamic sampling + entropy bonus) x 2 seeds, matched
compute (24 updates/run) so the comparison isn't confounded by training length.

**TASK:** Does a bundle of known GRPO-collapse mitigations (DAPO-style
dynamic sampling, entropy bonus, full task coverage — see `train_grpo_suite.py`
flags `--full_task_coverage --dynamic_sampling --entropy_coef`), applied for
the first time to a flow-matching VLA action head at <500M scale, beat vanilla
GRPO on standard LIBERO?

**COMMAND:** `run_campaign_worker.sh {0,1,2,3} 4`, 4 parallel workers on this
machine (the Blackwell node itself), launched 2026-08-30 03:15 KST, completed
2026-08-31 13:23 KST (~34h wall-clock, ~140 GPU-hours across 4 parallel workers).

**SOURCE:** `baseline_results/rl_{suite}_{cond}_s{seed}/rl_vs_baseline.json`
for all 16 runs; baselines from `baseline_results/{suite}_seed0_v2/summary.json`
(spatial: 3-seed mean from `baseline_results/summary.json`).

## RESULT — vanilla vs. baseline, fixed vs. baseline (n=200/arm/suite, 2 seeds)

| Suite | Baseline | Vanilla | Fixed | Winner (vs baseline) |
|---|---|---|---|---|
| Spatial | 77.33% | 76.50% (-0.83pp) | 78.50% (+1.17pp) | fixed |
| Object | 86.00% | 87.00% (+1.00pp) | 90.50% (+4.50pp) | fixed |
| Goal | 93.00% | 90.50% (-2.50pp) | 89.50% (-3.50pp) | vanilla |
| Long | 55.00% | 61.50% (+6.50pp) | 68.50% (+13.50pp) | fixed |
| **Pooled (n=800/arm)** | 77.75% | 78.88% (+1.12pp) | 81.75% (+4.00pp) | fixed |

## RESULT — fixed vs. vanilla head-to-head (the actual research question)

Two-proportion z-test, n=200/arm/suite (n=800/arm pooled):

| Suite | Vanilla | Fixed | Delta | z | p |
|---|---|---|---|---|---|
| Spatial | 76.5% | 78.5% | +2.00pp | 0.48 | 0.632 |
| Object | 87.0% | 90.5% | +3.50pp | 1.11 | 0.268 |
| Goal | 90.5% | 89.5% | **-1.00pp** | -0.33 | 0.739 |
| Long | 61.5% | 68.5% | +7.00pp | 1.47 | 0.142 |
| **Pooled** | 78.88% | 81.75% | **+2.88pp** | 1.45 | **0.148** |

**CHECK:** re-derived directly from the 16 `rl_vs_baseline.json` files (not
hand-copied); z-test recomputed via stdlib `math.erfc`, no manual arithmetic
in the final table. Script inline, reproducible from this file's command block.

## Honest verdict — 4-question rubric (`paper-evaluation.md`)

1. **REAL?** No. Pooled p=0.148, n=2 seeds/suite. No suite individually
   significant. Not distinguishable from noise on this evidence.
2. **CAUSED by the mechanism?** Not established. Three changes (full task
   coverage, dynamic sampling, entropy bonus) bundled with no ablation
   isolating any one — cannot attribute the effect, or rule out "just more
   gradient steps."
3. **NOVEL?** Weak pass. Individual components (DAPO-style dynamic sampling,
   entropy bonus) are known LLM-RL techniques; novelty is only "first ported
   to a flow-matching VLA head at this scale" — a legitimate but thin twist
   per the novelty gate, further weakened by (4).
4. **MATTERS?** No. Wins 3 of 4 suites but **loses on Goal — the near-ceiling
   suite the whole mitigation was motivated by**. That's not a narrow-scope
   caveat, it's the mechanism failing its own test case.

**Verdict: REVISE bordering KILL as a numeric/method claim.** Same category
as the original vanilla-GRPO result (PHASE4): directionally positive
(+2.88pp pooled, wins 3/4 suites) but not statistically real, not causally
isolated, and fails on its own motivating case. Not CVPR-caliber at this
evidence level — see conversation record for the full rubric defense.

What's honestly publishable from this: a workshop-scale or technical-report
finding — *"known GRPO collapse mitigations, ported to flow-matching VLA
policies, produce a small positive pooled effect (+2.88pp, n=800/arm,
p=0.148) but fail specifically on the near-ceiling suite they were intended
to rescue."* A mechanism/negative-result note, not a performance claim.

## What would be needed to go further

- Ablation isolating each of the 3 changes (or at minimum: mitigations-on vs.
  full-task-coverage-only vs. vanilla) to attribute the effect.
- Real seed count (~15-16/arm, previously power-analyzed) for significance.
- An explanation for why Goal (near-ceiling) got worse under the fix while
  Object/Long (more headroom) improved — is the "ceiling" framing itself
  wrong, or does the fix need suite-specific tuning?

## Files

- `run_campaign.sh`, `run_campaign_worker.sh` — orchestration (4-worker parallel)
- `train_grpo_suite.py` — `--full_task_coverage --dynamic_sampling --entropy_coef` flags
- `logs/campaign_manifest.jsonl` — full run log, all 16 TRAIN_END/EVAL_END/WORKER_DONE events
- `baseline_results/rl_libero_{spatial,object,goal,10}_{vanilla,fixed}_s{0,1}/` — all 16 raw results
