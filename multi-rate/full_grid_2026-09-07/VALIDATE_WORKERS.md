# Validation — `harness.py --workers N` correctness

Written 2026-09-15 11:28 KST. Before `--workers > 1` is used for any real campaign, this test
must pass: `--workers 1` and `--workers N` must produce byte-identical per-episode success arrays
on the same task/checkpoint/seed, not just matching aggregate stats (an aggregate match at small n
can hide a per-episode reordering bug if enough arms happen to sum equally).

## Why it's safe by construction

`run_episode`'s only source of randomness is `torch.manual_seed(1_000_003 * ei + replan)` — a pure
function of `(episode_index, replan_count)`, independent of arm identity and independent of which
process/worker executes it. Each worker builds its own `sim` and `policy` once (`_worker_init`);
results are reassembled by sorting on episode index (`_eval_parallel`'s final loop), not completion
order, so wall-clock scheduling across workers cannot change which episode's result lands where.

## Test run (PushT-v1, DP, k=2, seed=0, dp_PushT-v1.pt, n_eval=8, arms=native,zoh,spline,qp,qp_anchor)

```
python harness.py PushT-v1 --k 2 --seed 0 --n_eval 8 --arms native,zoh,spline,qp,qp_anchor --workers 1
python harness.py PushT-v1 --k 2 --seed 0 --n_eval 8 --arms native,zoh,spline,qp,qp_anchor --workers 4
```

Per-episode arrays, diffed directly (not just aggregate %):

| arm | workers=1 | workers=4 | match |
|---|---|---|---|
| native | [F,T,F,T,T,F,F,F] | [F,T,F,T,T,F,F,F] | yes |
| zoh | [T,F,F,F,T,F,F,T] | [T,F,F,F,T,F,F,T] | yes |
| spline | [T,T,F,T,F,F,F,F] | [T,T,F,T,F,F,F,F] | yes |
| qp | [F,F,F,T,T,F,F,T] | [F,F,F,T,T,F,F,T] | yes |
| qp_anchor | [F,F,F,T,T,F,F,T] | [F,F,F,T,T,F,F,T] | yes |

All 5 arms byte-identical across all 8 episodes. Wall time: 292s (workers=1) vs 188s (workers=4) —
only ~1.6x at this tiny n (fixed per-worker setup cost — spawning + model load x4 — dominates at
n=8; the win scales with n, since setup is paid once per worker, not once per episode).

**Verdict: validated. Safe to use `--workers N` for real campaigns going forward.**

## Known limitation, not yet tested

`RoboCasaSim` talks to `robocasa_bridge.py` over one socket per client; the bridge log has shown
multiple concurrent client connections before, but `--workers > 1` on a RoboCasa task has not been
explicitly validated the same way as ManiSkill above. Run the same byte-identical check before
trusting it there.
