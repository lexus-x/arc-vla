# Pre-registration — Gym Push-T absolute-position resampling

Written 2026-09-16 KST before any resampler result from the official Hub checkpoint exists.
The prior native-only evaluation is known (93.2% mean coverage, 68% binary success).

## Method correction

Gym Push-T predicts absolute 2-D targets. Applying the delta-action resamplers directly to those
positions is invalid. For every predicted chunk, this evaluation anchors the path at the current
agent position, differences absolute targets into displacements, coarsens/resamples those
displacements, and integrates back to absolute targets. TAC-Fold, cubic spline, and ZOH therefore
preserve every coarse block endpoint. The existing delta-action QP/C-TAC is excluded because its
`[-1,1]` constraint applies to per-step deltas, not absolute targets. Before the full run, a
`ctac_position` arm was added: it keeps C-TAC's TAC-Fold-anchored objective and exact block sums,
but constrains `current_position + cumulative_displacement` to `[-1,1]`. This is the valid
absolute-position analogue. Satfix remains excluded because no corresponding position-bound
definition exists.

## Protocol

- Policy: official `lerobot/diffusion_pusht` checkpoint already stored locally.
- Environment: gym-pusht, 10 Hz, 300-step horizon.
- Paired seeds: 1000–1099; n=100; vector batch size 10.
- Common diffusion noise: deterministic stream reset for every vectorized rollout batch.
- Rate: k=2.
- Arms: native, ZOH, raw cubic spline, raw TAC-Fold, position-constrained C-TAC, raw B-spline
  Algorithm 1 (`eps=.005`).
- Primary metric: per-episode maximum coverage reward, matching the paper.
- Secondary metric: binary success.

## Hypotheses and analysis

Primary family (Holm m=4): TAC-Fold and position-constrained C-TAC, each versus raw cubic spline
and raw B-spline on paired maximum coverage, two-sided Wilcoxon signed-rank. Report paired mean
deltas and 95% paired bootstrap CIs.
Binary success uses exact paired McNemar as a secondary diagnostic. ZOH and native are reported
fully but are not added post-hoc to the primary family.

No arm, seed, or metric may be dropped after seeing results. A tie or loss is reported as measured.
