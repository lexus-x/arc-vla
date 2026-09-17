# First principles: why interpolant choice does (and doesn't) matter for VLA action resampling

Synthesis across every campaign in this project (2026-08-15 → 2026-09-07). No new training
runs; the one new check below (B-spline's numerical failure mode) reuses real trajectory data
already on disk and took under a minute to run.

## 1. The mechanism that's actually doing the work

A robot controller like `pd_joint_delta_pos` (ManiSkill) or an OSC-POSE controller
(RoboMimic/RoboCasa) clips every per-step action to a fixed range, usually `[-1, 1]`. When you
resample a k-step action chunk to a different rate, the only thing that matters for task success
is whether the **resampled chunk's per-block sum equals the original block's sum** (the
displacement the demonstration or policy actually intended) *and* whether the individual
sub-steps stay inside `[-1, 1]` so nothing gets clipped and silently lost.

- **ZOH** (split the block sum evenly across k sub-steps) satisfies both by construction. It's
  the reference every other method is chasing, not a naive baseline.
- **satfix** is the general fix: after ANY interpolant runs, project each block back onto
  `{sum = S, |v_i| ≤ 1}`. This is why satfix helps unconstrained cubic spline a lot (spline
  overshoots the most, so there's the most to fix) and helps TAC-Fold only a little (TAC-Fold's
  own slope-limiting already avoids most overshoot).
- **TAC-Fold, plain cubic spline, and PCHIP are the same family.** All three are *interpolating*
  Hermite/polynomial splines — by construction they pass exactly through every coarse-block
  boundary, so their per-block sum is always exactly right even before satfix runs (verified
  below: measured per-block sum error is 0.00000 for all three on real PushT data). They differ
  only in how they estimate the tangent/slope at each knot (unconstrained cubic vs Fritsch-Carlson
  monotone vs Akima weighted-harmonic-mean), which changes how much they overshoot *between*
  knots, not whether they preserve the total. This is why no amount of searching within this
  family produces a "TAC-Fold beats everything" result — they're variations on one mechanism, not
  three different ideas. Diagnostic confirmation already on record (2026-09-06): reconstruction
  MSE is statistically indistinguishable across the three on 4/4 datasets tested.

## 2. Why closed-loop mostly erases this, except at precision moments

Every open-loop finding (no policy, static replay of pre-recorded actions) shows strong,
reproducible, Holm-significant interpolant effects. Almost none of that survives once a real
policy is in the loop and replanning. The reason: closed-loop replanning is itself a correction
mechanism. If block *i*'s resampled trajectory drifts slightly off the intended path, the policy
observes the *actual* resulting state next replan and corrects for it — the interpolation error
doesn't accumulate across the whole episode the way it does in open-loop replay. This is why the
2026-09-05 LIBERO/RoboCasa+GR00T closed-loop satfix runs were flatly null, and why today's
RoboMimic and RoboCasa results (near-ceiling or floor, no separation between arms) are null too:
these are low-precision, forgiving tasks where a slightly-off action gets absorbed by the next
correction. ManiSkill's PushT and PickCube are the exception, because they involve moments (fine
contact alignment, precise pushing) where a single bad chunk is enough to fail irrecoverably
before the next replan can fix it — that's specifically where this campaign's real signal lives.

## 3. Why the benchmark family matters as much as the resampler

Human teleoperation demonstrations (RoboMimic, RoboCasa, LIBERO) are low-bandwidth — a human
hand doesn't move at more than roughly 1 Hz of meaningful signal. There's almost nothing above
that frequency for any interpolant to get wrong, so ZOH, spline, PCHIP, TAC-Fold, and (normally)
B-spline are all statistically indistinguishable on these benchmarks. RL-generated demonstrations
(ManiSkill PushT, PickCube) are jerkier and routinely saturate the controller pre-clip (7-27% of
raw elements, measured this campaign) — that's real high-frequency content and real headroom for
resamplers to differ on, and it's the only place any of this project's interpolant comparisons
have ever produced a reproducible effect.

## 4. The corrected mechanism behind B-spline's collapse — verified, not a family property

The 2026-09-07 campaign reported B-spline catastrophically failing on PushT (3-4% success vs
26-30% native) and PickCube (44-47% vs 89-90% native), and attributed it loosely to "smoothing
splines trading exactness for smoothness." That explanation was incomplete. Direct measurement
on real PushT demonstration data (30 episodes, `scipy.interpolate.splprep`/`splev`, the exact
call `resample_bspline` makes) shows the real mechanism:

**Per-block sum reconstruction error, before satfix, mean over 30 real episodes:**

| resampler | mean per-block-sum error |
|---|---|
| spline | 0.00000 |
| pchip | 0.00000 |
| tac_fold | 0.00000 |
| **bspline** | **48,616** (dominated by outliers; typical-episode error is 2-4, itself already large) |

Tracing the single worst case (episode 43, k=2): five of six action dimensions fit within a
reasonable range of the true trajectory. The sixth blows up to a reconstructed value of
**830 million** against a true range of `[0, 13]`. This is `scipy.interpolate.splprep`'s
automatic-knot-placement smoothing-spline solver becoming numerically unstable — not a duplicate-
point degeneracy (the underlying cumulative trajectory is 51 distinct, smoothly-varying values),
but the FITPACK knot solver placing knots too tightly under the fixed absolute tolerance
`s=0.01`, producing Runge-phenomenon-style oscillation between knots on this dimension's specific
curvature. This is a known fragility of `splprep`'s automatic-knot smoothing mode on tightly
oscillating data, not a property of B-splines as a mathematical family — a genuine B-spline
*interpolant* (`scipy.interpolate.make_interp_spline`, or `splprep` with `s=0`) doesn't have this
failure mode, because it isn't solving an ill-conditioned smoothing optimization at all.

**Revised conclusion:** B-spline's closed-loop collapse is real (the eval numbers are correct),
but the mechanistic story is "this specific smoothing-spline implementation is numerically
fragile on saturating, oscillating action data," not "smoothing splines are unsuitable for
high-bandwidth robot actions." The two claims license different follow-ups — the first says fix
the implementation and re-test; the second would say abandon the whole method family. Don't
conflate them.

## 5. What this actually licenses (apply)

- Don't keep searching within {TAC-Fold, spline, PCHIP} for a version that "wins" — they're one
  mechanism with cosmetic differences, and three independent campaigns (09-05, 09-06, 09-07) have
  already shown the ceiling. This is settled, not under-explored.
- Don't cite "B-spline fails" as evidence against smoothing methods in general without the
  caveat above — the failure is substantially implementation-specific. If B-spline is worth
  testing again, first swap `splprep`(`s=0.01`) for either `s=0` (true interpolation) or
  `make_interp_spline`, and re-run only PushT/PickCube (the two tasks where it matters) before
  drawing any conclusion.
- The one closed-loop finding this whole project has that's real, reproducible, and mechanistically
  understood end-to-end is: **satfix's block-projection genuinely helps unconstrained cubic
  spline** specifically on saturating benchmarks (open-loop, robustly; closed-loop, only at
  precision-critical moments like PushT). That is the actual contribution, not "TAC-Fold wins" or
  "B-spline is bad."

## 6. Addendum (2026-09-08) — the correct B-spline arm, measured
Section 4 said the B-spline collapse was an implementation artifact. Confirmed at n=993: a
bounded-error B-spline resampler implemented per the B-spline Policy paper's Algorithm 1 (with the
numerical guards it needs on jerky data) scores 77.0% at k=2 on PickCube — tied with TAC-Fold+satfix
(77.3%), above cubic+satfix (72.8%), and above our QP (74.2%). At k=4 QP is best (35.2% vs
B-spline 29.6%, cubic 31.0%, TAC-Fold 30.0%). So the honest ranking is depth-dependent: shallow
decimation favours the damped/least-squares curve families; deep decimation favours the globally
constrained fit. Also: a B-spline interpolant is the cubic spline (identical to 5 digits), and the
reconstruction-MSE proxy mis-ranks the smoothing B-spline (predicts below cubic; success says above at
k=2) — MSE ranking is only trustworthy within the interpolant family.
