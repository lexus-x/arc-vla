# Phase 3 Final Result: RL Pilot vs. Reproduced Baseline (LIBERO-Spatial)

Preliminary, 1-seed existence-proof result. Not a significance claim. New chapter
within `multi-rate/vla-vault`, kept outside the vault's protected `output/`/`wiki/`.

## Task 4 — Final eval of the RL-trained checkpoint

**TASK:** Evaluate `rl_checkpoint_pilot/` (8-update GRPO pilot, starting from the
`flow8300_s0` SFT checkpoint) on LIBERO-Spatial, same protocol as the 3-seed
baseline reproduction (10 trials/task, seed 0), and compare against both the
same-seed baseline and the 3-seed baseline mean±std.

**COMMAND:** `conda run -n vla_smolvla_libero python rl_eval.py` (in
`/home/user/Desktop/multi-rate/vla-rft/`), writing
`baseline_results/rl_pilot_seed0/multirate_honest.json` and
`rl_eval_result.json`.

**SOURCE:** `/home/user/Desktop/multi-rate/vla-rft/rl_eval_result.json`,
cross-referenced against `baseline_results/{seed0,seed1,seed2}/multirate_honest.json`
and `baseline_results/summary.json` (Phase 2's 3-seed reproduction).

**RESULT (n=100 rollouts, seed 0, 10 trials/task × 10 tasks):**

| Arm | Success rate | n |
|---|---|---|
| RL pilot checkpoint (8 GRPO updates) | **80.00%** (80/100) | 100 |
| Baseline, same seed (seed 0) | 72.00% (72/100) | 100 |
| Baseline, 3-seed mean ± std | 77.33% ± 5.03% | 300 (100×3 seeds) |

- Delta vs. baseline **same-seed** (seed 0): **+8.00 pp**
- Delta vs. baseline **3-seed mean**: **+2.67 pp**
- Noise floor (2× the 3-seed baseline stddev, per this project's own standing
  rule): **10.07 pp**
- **Clears noise floor vs. mean: NO** (2.67pp < 10.07pp)
- **Clears noise floor vs. same-seed baseline: NO** (8.00pp < 10.07pp)

**CHECK:** `python -c "import json; d=json.load(open('rl_eval_result.json')); assert d['clears_noise_floor_vs_mean']==False; print('confirmed: does not clear noise floor')"` — executed, passes clean.

## Honest verdict

Both measured deltas are **positive** (RL checkpoint scored higher than the
baseline on both comparisons available), but **neither clears this project's
own noise-floor bar** established from our own 3-seed reproduction (77.33% ±
5.03%, so ~10pp needed to be distinguishable from seed-to-seed noise). This is
a **preliminary, inconclusive-but-not-negative** result, not a positive
finding — it must not be reported as "RL post-training improves LIBERO-Spatial
performance" on this evidence alone.

What this pilot *does* establish, which is real and load-bearing regardless of
the eval outcome:
1. **Mechanistic existence proof**: the ported Flow-SDE mechanism produces
   finite log-probs, correct gradients, and real (non-degenerate) GRPO update
   steps on SmolVLA's flow-matching action head at <500M scale — 4 of 8 pilot
   updates had genuine gradient signal (loss 0.40–0.86, grad norms 139–168
   pre-clip). This was the open technical question from the novelty gate
   (does RL-on-flow-matching machinery, previously only shown at ~3B scale via
   π_RL, work at all at SmolVLA's scale) — mechanically, yes.
2. **A concrete, reproducible noise floor** for this exact checkpoint/protocol
   (77.33% ± 5.03%, n=100/seed × 3 seeds) that any future run against this
   checkpoint should be judged against.
3. **A real, documented GRPO-in-this-regime limitation**: half the pilot's
   updates were degenerate (zero within-group reward variance) at
   group_size=6, given SmolVLA's already-high/already-low per-task success
   rates leaving little room for a mixed group — worth stating explicitly in
   any writeup's limitations section.

## Implication for the full-rigor follow-up

This result is **directionally encouraging but not sufficient to justify the
full 2.5–4 week multi-suite, multi-seed investment on its own** — it does not
provide evidence against continuing (both deltas are positive, mechanism
works), but it also doesn't provide evidence for a real effect yet (neither
delta is significant). The honest next step, if pursuing further, is more
seeds and more updates on this one suite before expanding scope, not a scope
expansion based on this pilot alone.

## Explicitly unresolved / not verified

- Whether a longer run (more than 8 updates) shows a trend — this pilot was
  intentionally short (existence-proof scope).
- McNemar/paired significance test on the seed-0 same-init-state comparison
  was not computed (only aggregate rates) — a genuine gap if a tighter
  statistical read is wanted before deciding on the follow-up.
- Whether the degenerate-update rate (4/8) would improve with a larger
  group_size — not tested.

## Files

- `/home/user/Desktop/multi-rate/vla-rft/rl_eval.py` (eval script)
- `/home/user/Desktop/multi-rate/vla-rft/rl_eval_result.json` (comparison summary)
- `/home/user/Desktop/multi-rate/vla-rft/baseline_results/rl_pilot_seed0/multirate_honest.json` (raw per-rollout data)
