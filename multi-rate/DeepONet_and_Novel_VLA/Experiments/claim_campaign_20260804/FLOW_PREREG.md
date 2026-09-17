# PREREG — DeepONet head vs flow matching, matched budget (written before any result)

Registered 2026-08-12, before any arm below was run. Every number these arms produce is
reported, in every direction.

## Why this comparison exists

Cubic spline resamples **our own policy's** chunk. A better decoder is therefore capped at the
0.75% interpolation error measured offline before any arm was built, and a better policy lifts
both arms equally. That comparison is structurally closed and it produced a tie
(Spatial n=300: -0.3 pp, p=1.0000; Goal n=300: -1.3 pp, p=0.5034).

Flow matching is a **different policy**, not a resampler of ours. Two things follow:
1. The comparison is not capped by an interpolation bound.
2. Flow has no operator head, no velocity field and no `dt`. It **cannot fold**. At an
   off-native rate its only options are resampling or holding.

## Arms

Budget-matched: every checkpoint is 8,300 steps (1,650 + 6,650), `libero_spatial_image`,
seed 0. Same evaluator, same wall-clock budget, same pinned initial states, same `replan`.

| Arm | Checkpoint | Rate | Decode |
|---|---|---|---|
| `asrc_native_20env` | `asrc_s0/8300` | 20 | native |
| `don_v2_native_20env` | `m3_deeponet_s0/8300` | 20 | native |
| `flow_m1_native_20env` | `m1_flow_s0/8300` | 20 | native |
| `asrc_cadmag_folding_40env` | `asrc_s0/8300` | 40 | **folding** |
| `flow_m1_cadmag_spline_40env` | `m1_flow_s0/8300` | 40 | spline (flow's best available) |

## Hypotheses and margins, fixed now

- **H1 (20 Hz, head quality).** `asrc` vs `flow_m1`, paired, n=300. Two-sided exact McNemar,
  alpha=0.05. No margin: this is a superiority test in either direction.
- **H2 (40 Hz, deployed).** `asrc`+folding vs `flow_m1`+spline, paired, n=300, same test.
- **H3 (decomposition).** If H2 is significant, H1 says how much of it is head quality rather
  than rate handling. **A win at 40 Hz that is fully explained by H1 is a head result, not a
  multi-rate result, and must be reported as one.**

## Kill / honesty rules

- n=300 per arm. n=50 is not reportable: this project has watched **3 of 3** n=50 estimates
  reverse under power, most recently Goal (+6.0 pp -> -1.3 pp).
- Report the **overall** number. Per-category or per-task cherry-picking is what invalidated
  the earlier "+33.4pp" claim; it will not be repeated.
- `flow_m1` gets the identical cadence + magscale fixes. Running flow without them would be a
  rigged comparison, since those fixes are worth +54 pp on their own.
- If flow wins, that is the result and it goes in the same table.
- The spline tie stands regardless of what these arms show. It is not superseded by them.
