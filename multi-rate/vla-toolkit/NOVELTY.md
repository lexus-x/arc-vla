# NOVELTY gate — Progress-Gated Adaptive Denoising (PGAD)

## Claim (one sentence)
We propose to gate the flow-matching sampler's per-chunk noise level using a
frozen, self-supervised dense-progress estimate, in order to reduce erratic
low-confidence actions specifically when the rollout appears stalled, which
existing inference-time correction methods do not do because they gate a
separate correction/injection module (or don't gate at all) rather than tuning
the base sampler's own stochasticity — a mechanism-level distinction from
every close paper found.

## Taxonomy placement
Axis E (inference-time mechanism, no retraining) — specifically "adaptive
compute," a sub-axis the taxonomy reference explicitly flags as
under-explored for small VLAs. Known point on this axis (test-time
search/best-of-N with a critic) is occupied by VLA-ATTC and siblings; our
point (adaptive *sampling stochasticity*, not candidate selection) is not.

## Why best-of-N / critic-based selection was rejected as the mechanism
The existing frozen asset (`progress_estimator.py`) is a pure *observation*
regressor — predict_progress(features, state) has no action-conditioning. It
cannot score candidate action chunks (no dependency on the action itself), so
it cannot drive a best-of-N selection the way ATTC's or Look-Before-You-Leap's
Q-critics do. Building an action-conditioned critic from scratch was ruled
out on time (2-day deadline) — training one properly is a multi-day project,
not an inference-time knob. Adaptive-stochasticity sidesteps this: the gate
only needs a state-based confidence signal, which is exactly what's on hand.

## Prior art (2026-09-03 search, 3 parallel agents, see conversation log)
- **ATTC** (2605.01194): gates a correction module for *compute*, not safety;
  ablation shows no regression. Different target for gating, different gated
  object (external module vs. own sampler).
- **RTCF** (2608.04527): documents real regression from always-on correction.
  Neither paper reconciles the other.
- **Guided Action Flow** (2607.02092): closest miss — critic-ensemble
  disagreement gates a correction module. Still gates an external module, not
  the sampler's own noise schedule; authors call their gate incomplete.
- **VLA-Corrector** (2607.01804), **Look Before You Leap** (2607.03751),
  **BOKBO** (2605.30660), **VLAConf** (2605.29605): all gate/monitor via
  classifier confidence, latent drift, or Q-ensemble disagreement — none use a
  dense self-supervised progress *regression* signal, and none tune sampler
  stochasticity as the intervention.
- **Realtime-VLA FLASH** (2605.13778) and the Spec-VLA/KERV/WA-SpecDec line:
  same taxonomy axis (E) but a different sub-axis (speculative decoding for
  speed) — orthogonal, not a competitor for this mechanism.

## Distinguishing twist (Step 4)
Legitimate twist per the checklist: "a mechanism known elsewhere (adaptive
compute / adaptive noise scheduling exists broadly in diffusion literature)
shown for the first time to work in a regime/role it wasn't tried in" — gating
a flow-matching *policy's own sampling stochasticity* by a dense progress
signal, specifically to reconcile the ATTC/RTCF contradiction, is that regime.
The mechanism never overrides or replaces an action (unlike every correction
method above), which is the structural reason it should not regress
near-ceiling tasks the way an injected correction can.

## Contribution test (Step 5)
Honest answer: **a tweak, not a new mechanism class** — adaptive sampling
schedules are known; the contribution is the trigger signal (self-supervised
progress regression) and the specific target (resolving a named, unresolved
contradiction in 2026 VLA literature), not a new sampling algorithm. This
matches the user's explicit bar: novelty 6-10/10, not a "new brain." Per the
checklist, a tweak needs a bigger/cleaner empirical win to justify itself —
this is why BASELINE reuse + a clean 3-arm ablation (frozen / always-careful /
gated) matters more here than for a higher-novelty candidate.

## Verdict: SURVIVES
No prior art occupies this exact point (progress-regression-gated sampler
stochasticity). Novelty is deliberately modest (6-10 target) per user spec;
everything downstream (BASELINE reuse, ablation, stats) needs to be as clean
as possible to compensate, per the checklist's own guidance on tweaks.
