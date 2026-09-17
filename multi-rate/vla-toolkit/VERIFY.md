# VERIFY — PGAD, first campaign result

n=20 episodes/task, seed-paired against the existing (not rerun) frozen
baseline. 8 tasks, 4 LIBERO suites, SmolVLA <500M, gated vs. always_careful
vs. frozen.

## Results (see pgad_stats.py output, 2026-09-03)

| check | frozen | variant | delta | p |
|---|---|---|---|---|
| Ceiling tasks (spatial-7, goal-1), gated | 39/40=97.5% | 38/40=95.0% | -2.5pp | 0.556 |
| Weak tasks (6 tasks), gated | 84/120=70.0% | 79/120=65.8% | -4.17pp | 0.489 |
| Pooled, gated | 123/160=76.9% | 117/160=73.1% | -3.75pp | 0.439 |
| Pooled, McNemar (paired) | b=21, c=15 | — | net -3.75pp | 0.405 |
| Pooled, always_careful | 123/160=76.9% | 111/160=69.4% | -7.5pp | 0.130 |

## Honest reading
No effect clears significance in **either direction**. Where there is a
trend, it is **negative, not positive**: gated is directionally worse than
frozen pooled and on the weak-task bucket (the bucket the mechanism was
supposed to help); always_careful is worse still, and closest of the three to
significance. Per-task heterogeneity is large and sign-inconsistent (e.g.
libero_10 task0 drops 7/20→3/20 under gating, while libero_10 task1 improves
14/20→16/20) — the same noisy, sign-flipping pattern this project has hit
before (candidate F), not a stable directional effect.

**The important diagnostic finding:** always_careful's larger (though still
non-significant) negative delta suggests the mechanism's core premise —
"reduce flow-SDE noise_level to be more careful when progress stalls" — may
be backwards. Lower noise_level makes the sampler more deterministic, which
could just as plausibly *trap* the policy in a locally-bad action distribution
as protect it from one; the data leans toward the former. This is a genuine,
usable finding, not just "not significant."

## Rubric (NOVEL / REAL / CAUSED / MATTERS)
- **NOVEL**: still holds — nothing found in prior art gates flow-SDE noise
  by a dense progress-regression signal. Unaffected by this result.
- **REAL**: FAILS. No direction clears the noise band; where there's a
  trend it's negative.
- **CAUSED**: partial pass, negative-direction — the always_careful arm
  isolates that lowering noise_level itself is the likely driver of any
  effect, and it drives performance down, not up.
- **MATTERS**: FAILS as a performance claim. As a diagnostic ("naive
  noise-reduction is not free lunch for stalled VLA rollouts") it's a real
  but modest finding — not the "barely enough to get accepted, novelty 6-10,
  everything else 9-10" performance paper the user asked for.

## Verdict: REVISE, not KILL
The infrastructure, progress-heads, gate logic, and seed-paired eval design
are all sound and reusable — this specific choice of intervention (lower
noise on stall) is the thing that failed, not the framing. Cheapest real fix:
**test the opposite sign** — raise noise_level (more exploration/stochasticity)
on a detected stall instead of lowering it, on the hypothesis that a stalled
rollout needs to escape a bad mode, not be dampened into it. This requires
changing one constant (`NOISE_CAREFUL` -> `NOISE_EXPLORE`, e.g. 0.25-0.3) and
rerunning; all other infra (progress heads, gate class, eval harness, stats)
is unchanged. Given the deadline, a fast partial check (the 2 tasks with the
largest observed effect either direction: libero_10 task0 and spatial task8)
before committing to a full rerun is the pragmatic next step.
