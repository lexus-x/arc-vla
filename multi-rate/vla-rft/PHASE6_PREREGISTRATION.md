# PHASE6 Pre-registration: candidate F ("predict per-task RL benefit before training")

Committed 2026-09-01T09:17:47+09:00 (probe completion timestamp, `logs/phase6_probe_results.json`),
**before** any of PHASE6's full GRPO/PPO training runs start. Per this project's own
discipline (see `multi_task_campaign.py`'s docstring): a degenerate or wrong prediction
on any task is reported as-is, not swapped out or re-explained after the fact.

## Task grid (8 tasks, 4 suites, none previously isolated-eval'd in this project)

Verified against every existing `logs/campaign_*.json`, `baseline_results/*/*.json`, and
the `multi_task_campaign.py` module docstring's own documented task history before
selection — see that file's `SUITE_TASK_IDS` PHASE6 addendum comment for the exclusion
reasoning per suite.

## Prediction rule (from `calibration_probe.py`, unchanged from its pre-run form)

```
baseline_success_rate >= 0.90  -> REGRESS_OR_FLAT   (near-ceiling; Qwen-VLA arXiv:2605.30280
                                                       precedent: ~97.8% baseline -> +0.1pp RL gain)
baseline_success_rate <= 0.10  -> REGRESS_OR_FLAT   (near-floor; same degenerate-advantage
                                                       mechanism, opposite end)
degenerate_group_rate >= 0.5   -> REGRESS_OR_FLAT   (PHASE5 Long-suite precedent: 5/8=62.5%
                                                       degenerate GRPO updates coincided with
                                                       that suite's regression)
otherwise                      -> IMPROVE
```

Probe: 3 groups x 4 rollouts = 12 frozen-policy episodes/task (vs 20 for a full eval arm),
same GROUP_SIZE as the GRPO/PPO arms so `degenerate_group_rate` is measured the same way
training will encounter it.

## Committed predictions

| suite | task_id | baseline_success_rate | degenerate_group_rate | **PREDICTION** |
|---|---|---|---|---|
| libero_spatial | 7 | 1.000 | 1.000 | **REGRESS_OR_FLAT** |
| libero_spatial | 8 | 0.833 | 0.333 | **IMPROVE** |
| libero_object  | 0 | 0.917 | 0.667 | **REGRESS_OR_FLAT** |
| libero_object  | 1 | 0.500 | 0.000 | **IMPROVE** |
| libero_goal    | 0 | 0.750 | 0.333 | **IMPROVE** |
| libero_goal    | 1 | 1.000 | 1.000 | **REGRESS_OR_FLAT** |
| libero_10      | 0 | 0.250 | 0.333 | **IMPROVE** |
| libero_10      | 1 | 0.750 | 0.333 | **IMPROVE** |

3 REGRESS_OR_FLAT, 5 IMPROVE — not degenerate (probe didn't just predict one class for
everything), so this is a meaningful test of the rule, not a foregone conclusion either way.

## What "success" for candidate F means

For each task, compare the prediction above against the actual GRPO-arm outcome from
PHASE6's full run (`logs/campaign_{suite}_task{task_id}_grpo.json` vs the paired frozen
arm, same delta_pp/two-proportion-z pattern as `multi_task_report.py`). A task counts as
a **hit** if: predicted IMPROVE and actual delta_pp > 0, or predicted REGRESS_OR_FLAT and
actual delta_pp <= 0. Report the hit rate across all 8 as-is — this file is the commitment
that fixes what "as-is" means before the numbers exist.

## Full run (launched immediately after this file was written)

`multi_task_campaign.py`'s `SUITE_TASK_IDS` extended with this exact 8-task grid (PHASE6
addendum, see that file). Launched via `run_multi_task_campaign.sh` (4 workers, same
precedent as the prior 21-run campaign) — existing completed (suite,task,arm) combos are
skipped automatically; only these 24 new combos (8 tasks x 3 arms) run.
