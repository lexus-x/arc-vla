# FRAME v2 — IEEE Robotics-tier target, longer runway

Status: 2026-09-04, supersedes FRAME.md's 2-day-deadline framing (that
deadline is over; this is a fresh cycle with real runway, days not hours).

## What changed from FRAME.md
- Venue: target IEEE Robotics tier (RA-L / ICRA / IROS / T-RO), not
  NeurIPS/CVPR-adjacent. Different reviewing culture: more receptive to
  systems/safety/robustness/benchmarking contributions with rigorous
  evaluation, less singularly focused on ML-theoretic novelty.
- Bar: user explicitly raised this from "barely enough, novelty 6-10" to
  "at least IEEE robotics level SIGNIFICANT" — the result itself must clear
  real significance, not just be publishable-adjacent on a technicality.
- Timeline: days, not hours. Can afford proper multi-seed campaigns and a
  careful IDEATE/NOVELTY pass instead of racing to the first idea that survives.

## Hard-won lessons from the killed session (do not repeat)
1. **"Gate a cheap inference-time knob using a confidence/progress signal" is
   a saturated axis for VLA as of 2026** — 3 independent variants (noise-level
   both directions, replan-cadence, denoising-step-count) either failed
   empirically or were already closed by prior art (CVPR 2026 paper on
   entropy-gated chunking, DVAC on variance-gated cadence, etc.). Do not
   propose a 4th variant of this same pattern.
2. **"Reshape GRPO's reward with the progress signal" also failed empirically**
   (candidate G, -5pp pooled, p=0.53, wrong direction both seeds). The
   progress-regression signal itself is accurate (r≈0.99 held-out) but nothing
   built on top of it to change policy behavior has helped yet, across 4 real
   empirical tests. Treat any new candidate that reuses this signal as an
   intervention trigger with real suspicion — the signal being accurate does
   not mean it's the right lever.
3. Fast novelty checks BEFORE implementation saved real time twice this
   session (replan-cadence, denoising-steps) — keep doing that, especially now
   with a broader IEEE-robotics prior-art sweep needed too (different venue,
   different closest-competitor set).

## New research question (draft, needs IDEATE pass before locking)
Given SmolVLA (<500M) on LIBERO (all 4 suites) + LIBERO-Plus (OOD robustness
suite, already wired from candidate H's work), what is a genuinely significant,
IEEE-Robotics-caliber contribution on an axis NOT yet touched this session:
architecture (A), action representation (B, though object-relative was already
killed as a "well-trodden trap"), data (D, candidate D's failure-mode
curation was left at "narrow daylight, not fully explored"), or a
robustness/safety framing native to IEEE robotics culture that this session
never made the PRIMARY target (LIBERO-Plus was only ever a secondary eval).

## Next stage
PRIOR ART, properly, before any more IDEATE: (1) what does "significant" and
"accepted" actually look like at IEEE Robotics venues for small-VLA work
recently, (2) re-check candidate D's current status, (3) survey the safety/
robustness axis as a possible primary target given the infra already on hand
(LIBERO-Plus, joint-velocity/jerk safety metrics from multi_task_campaign.py).
