# FRAME — cheap, general-purpose VLA toolkit paper

Status: proposed 2026-09-03, proceeding under auto-mode default (user said "no idea,
find the gap and fill" — treating this framing as accepted unless redirected).

## Target bar (explicit user spec)
- Novelty: 6-10/10 is fine — does NOT need a "new brain" architecture.
- Everything else (rigor, real effect, causal attribution, practical relevance): 9-10/10.
- Framing: addresses *multiple* problems with cheap, small fixes, applicable to
  *any* VLA backbone (not tied to one architecture).
- Venue: NeurIPS/CVPR-tier main track or adjacent (findings/workshop track
  acceptable) — not requiring a top-tier headline result. "Barely enough to get
  in, good enough for a PhD chapter."

## Working research question
Can a small, backbone-agnostic toolkit of cheap (<1% extra params, no backbone
retraining) plug-in interventions for pretrained VLA policies be combined into
one recipe that:
1. fixes a real, previously-unresolved failure mode of existing single-component
   fixes (candidate: near-ceiling regression under inference-time correction —
   ATTC says no regression, RTCF says yes, contradiction unresolved in lit), AND
2. adds a second, independent cheap win (candidate: inference-time efficiency
   via block-parallel/speculative action decoding — survey-confirmed open for
   VLA, arXiv:2608.20743), AND
3. transfers across ≥2 VLA backbones/action-head types without retraining the
   backbone?

## Registered success criteria (write BEFORE running anything)
- REAL: pooled improvement over frozen/naive baseline, p<0.05, reported with CI,
  OR an explicit non-regression guarantee with tight CI if the effect is a
  safety/floor property rather than a mean shift.
- CAUSED: each toolkit component ablated independently (on/off grid) — no
  bundling three unattributed changes again (repeat of the Collapse-Aware-GRPO
  mistake, see project memory).
- MATTERS: must resolve or explain the ATTC/RTCF contradiction, not just add
  another data point to it.
- GENERALITY: claim capped to what's actually tested. If only one backbone is
  feasible given compute/checkpoints on hand, say so as a disclosed limitation —
  do not oversell "any VLA" past the evidence.
- Reuse existing repo assets where possible (progress_estimator.py,
  calibration_probe.py, multi_task_campaign.py infra) to keep cost cheap and
  execution rigor high — this is a ponytail-style "least code that works"
  constraint applied to the research plan itself, not just the code.

## Isolation
All new work lives in /home/user/Desktop/multi-rate/vla-toolkit/. Reads from
../vla-rft/ and ../vla-vault/ are fine (existing checkpoints, infra, prior
results); no writes outside this folder.

## Next stage
PRIOR ART — verify the specific *combination* (gated-dispatch generality fix +
speculative/parallel decoding, unified as one backbone-agnostic recipe) isn't
already published, and re-check current state of both components for anything
newer than the 2026-09-01 searches already on record.
