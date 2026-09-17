# IDEATE v2 — 3 candidates, IEEE-Robotics-tier target

Venue calibration (from prior-art sweep): 15-40pp recovery reads as a strong
headline; single-digit pp is acceptable only paired with an ID/OOD
gap-closure framing or multi-metric corroboration. Real robotics venues
expect multi-seed variance reporting at minimum. Small-model framing itself
is a valid contribution angle (TinyVLA, RA-L 2025) independent of raw SOTA.

## Candidate R1 — Dual-stream (vision + proprioception) robustness adapter
**Mechanism:** a small, cheap adapter/gating module that normalizes or
restores BOTH the vision stream AND the proprioceptive stream under
perturbation, for a frozen or lightly-tuned small (<500M) flow-matching VLA.
**Why now:** two independent searches converged on the same gap — (a) no
<500M-scale VLA robustness method exists at all (every BYOVLA/ROAD-VLA/
STRONG-VLA/RobustVLA target 7B-class OpenVLA/π0), and (b) vision-only
training-free fixes have a *measured* ceiling (G-PSMR: ~3pp on LIBERO-Plus,
can't fix proprioceptive-shift failures) — nobody has closed that specific
gap. Axis: A (architecture) / E (inference), touches both streams.
**Risk:** G-PSMR (MDPI Entropy 2026) is the closest, already does per-stream
gated restoration and already evaluates on LIBERO-Plus — need to verify
exactly what "streams" it restores and whether small-model-scale is really
untouched by it specifically, not just by the older 7B-class papers.

## Candidate D — Failure-mode-clustering-driven data curation (carried over)
**Mechanism:** cluster a frozen small VLA's LIBERO failures by MLLM-labeled
failure mode, use the taxonomy to targeted-curate/augment training data,
closed-loop retrain, measure the gain vs. uniform-scaling baseline.
**Why still live:** differentiation from RECALL (arXiv:2606.23617) reconfirmed
solid — RECALL uses pure statistical uncertainty, no semantic clustering.
Newer competitor (arXiv:2506.06570) does MLLM failure-taxonomy clustering but
never closes the loop with an actual VLA+LIBERO retrain — that specific
closed-loop combination remains unclaimed. IEEE framing explicitly confirmed
as a stronger fit than ML venues (data-efficiency angle matches RA-L/ICRA
reviewer priorities). Axis: D (data).
**Risk:** narrower daylight than R1 — needs sharp, explicit differentiation
from 2506.06570 in the writing, and is the most implementation-heavy (needs
an MLLM labeling pass + a retrain loop, not just an eval).

## Candidate R2 — Reliability-weighted test-time ensembling for flow-matching action chunks
**Mechanism:** draw K independent stochastic action-chunk samples (flow_sde.py
already supports this cheaply) and combine them via a SAFER-style
reliability/agreement weighting, instead of taking one sample.
**Why now:** SAFER (arXiv:2606.22351) is described as "directly transferable,
unexplored on VLA/LIBERO-Plus" — genuinely different mechanism from every
inference-time thing killed this session (those gated ONE sample's
parameters; this ensembles MULTIPLE independent samples).
**Risk — real and significant:** best-of-N-style action selection is
adjacent territory this project already found crowded (VLA-ATTC and
siblings). "Ensembling by reliability" vs. "select-by-critic" is a real
mechanism difference, but a skeptic could call it a thin variant of best-of-N
unless the reliability signal and combination rule are clearly distinct.
Needs the most adversarial NOVELTY pass of the three.

## Recommendation
R1 has the strongest, most independently-corroborated prior-art gap (two
separate search agents converged on it unprompted) and the clearest
IEEE-venue fit (BYOVLA is the direct template, ICRA 2025). D is a solid
second, already vetted twice. R2 is the highest-novelty-risk of the three
and needs a dedicated adversarial check before any implementation time goes
into it.
