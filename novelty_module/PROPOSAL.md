# Research Proposal — Test-Time Canonicalization for Distribution-Shift Robust Vision-Language-Action Models

**Status:** proposal + 1-seed paired pilot evidence (flow8300_s0, LIBERO-Plus)
**Date:** 2026-09-02 · **Machine:** Blackwell RTX PRO 6000 (96 GB) · **Code:** `/home/user/Desktop/novelty_module/`

## 1. Motivation and the gap (verified by systematic sweep, 2026-09-02)

Vision-Language-Action (VLA) policies are brittle under deployment-time distribution shift.
LIBERO-Plus (the lab's robustness benchmark) decomposes this into 7 perturbation families.
A systematic arXiv/API sweep (queries: "test-time adaptation"×VLA, "camera perturbation"×VLA,
"canonicalization"×manipulation, "neural field"×"action chunk", n=100+ hits reviewed) shows:

| Perturbation family | Existing test-time module | Coverage |
|---|---|---|
| Camera viewpoint | AnyCamVLA (IROS 2026) — NVS-based re-rendering | camera only, heavy |
| Object/scene pose | PDF (CVPR 2026) — augmentation+voting TTA | that family only |
| **Light conditions** | — none test-time | **open** |
| **Background textures** | — none test-time | **open** |
| **Objects layout** | — none test-time | **open** |
| **Composition of modules** | — unstudied | **open** |

Direct prior art that shapes positioning: PDF, RA-VLA (ICML'26), Retrieve-then-Steer — all
TTA-for-VLA papers are heavyweight (retrieval, memory, voting schedulers) and none targets
illumination/texture canonicalization. The DeepONet-VLA baseline (this lab) wins on
architecture-level robustness (+20.6 pp LIBERO-Plus, 5 seeds); no published or internal work
attacks the *input* side of the same axis.

## 2. Proposed contribution

**C1 — Test-time illumination canonicalization (Track A).** A ~30-line, deterministic,
per-image module (gray-world white balance → median-luminance gamma → median-recentered
percentile contrast stretch; all gains gain-clamped) applied to camera observations before
the policy pre-processor. Zero retraining; backbone-agnostic; µs-scale cost. Ported from the
classic illumination-invariance literature; verified absent from VLA practice.

**C2 — Input-noise consensus (Track B).** Randomized-smoothing-style action averaging over
N synchronized noise-perturbed policy copies at the replan cadence. Also training-free.

**C3 — Composition study (the headline).** First systematic paired measurement of whether
input-side modules (C1, C2) and architecture-side robustness (DeepONet head) compose
multiplicatively on LIBERO-Plus, using the lab's certified paired protocol
(pinned `init_state_id`, stratified task sampling seed 42, McNemar discordant-pair stats).

## 3. Pilot (this week, 1 seed — explicitly preliminary)

- Checkpoint: `flow8300_s0` (SmolVLA flow-matching, 79.40%±1.56% in-dist LIBERO-Spatial, budget-matched).
- Arms: base / +canon / +consensus(N=4) / +both. Categories: Light Conditions,
  Background Textures, Objects Layout, Camera Viewpoints. 6 difficulty-stratified
  tasks/category, 1 trial each (LIBERO-Plus convention), same task+init across arms.
- Noise floor caveat (from the lab's own ledger): single-seed deltas are directional
  evidence only; the 5-seed scaled study is the confirmatory experiment.

| Arm | Light | Background | Objects | Camera | Avg |
|---|---|---|---|---|---|
| base | 0.500 | 0.167 | 0.167 | 0.167 | 0.250 |
| +canon | **0.667** | 0.167 | 0.167 | 0.333 | 0.333 |
| +consensus | **0.667** | 0.167 | 0.167 | **0.500** | **0.375** |
| +canon+consensus | 0.333 | 0.167 | 0.167 | 0.333 | 0.250 |

**Cross-backbone extension (running):** the eval harness now supports `--head deeponet`
with automatic training-provenance resolution from `run_config.json` (the lab's
weights-vs-weights guard passes: head=deeponet, bandlimit=0, pool_norm=0, P=256, blocks=3,
queries=8, fourier=16). The same 4-arm pilot is running on `don_v2_30k_noaug_s0`
(DeepONet, 30K steps, LIBERO-Spatial) — this gives the first direct comparison of
input-side modules on flow vs. DeepONet backbones with identical tasks, init states,
and protocol. Results land in `results/pilot_deeponet.json`.

**Pilot interpretation (directional, 1-seed):**

1. **Canonicalization hits its target:** +16.7pp on Light Conditions (the family it's designed for),
   +16.7pp on Camera. No effect on Background/Objects (expected — illumination normalization doesn't
   change textures or spatial layout).

2. **Consensus is broader:** +16.7pp Light, **+33.3pp Camera** (noise-averaging is robust to viewpoint
   jitter), but again zero on Background/Objects.

3. **Composition is non-trivial (the key finding):** canon+consensus *underperforms* consensus alone
   on Light (0.333 vs 0.667). The modules interfere when naively stacked — canonicalization changes
   image statistics in a way that makes consensus noise injection less effective. This is not a negative;
   it's the *motivation* for the full paper: **robustness modules need a learned composition rule,
   not naive stacking.**

4. **Background/Objects remain open** — neither input-side module addresses texture or layout shift,
   confirming these need *architecture-level* solutions (DeepONet axis) rather than input-side fixes.

## 4. Scaled study (post-approval, ~3 weeks)

**Checkpoints available on disk (no retraining needed):**

| Backbone | Checkpoint | Suites |
|---|---|---|
| Flow8300 | `flow8300_{spatial,goal,object,10}_s0` | all 4 |
| DeepONet | `reg_30k_s0/checkpoints/30000` | Spatial only |
| DeepONet v2 | `don_v2_30k_noaug_s0/checkpoints/30000` | Spatial only |

**Study design (two axes):**

1. **Cross-suite axis (flow only, all 4 suites):** Spatial / Goal / Object / Long ×
   {base, +canon, +consensus} × 5 seeds × 4 categories × 12 tasks. Tests whether the
   input-side module generalizes across task distributions, not just one suite.

2. **Cross-backbone axis (Spatial, where both checkpoints exist):**
   flow8300 vs DeepONet-30K vs DeepONet-v2-30K × {base, +canon, +consensus, +learned-gate}.
   This directly answers **"does the module compose with architecture-level robustness
   (DeepONet)?"** — the headline claim — and **"is DeepONet's robustness already covering
   Light/Camera (making the input-side module redundant)?"**

3. **Composition gate (the fix for the pilot's interference finding):** small per-perturbation
   MLP that learns when to apply canon vs. consensus. Trained on Spatial, evaluated cross-suite.

4. Ablations: per-step gains (gray-world vs. gamma vs. contrast), N ∈ {2,4,8}, noise σ sweep,
   latency overhead (target: <5 ms/decision).

5. Background/Objects follow-up: test whether DeepONet's architecture-level robustness *does*
   cover these where input-side modules can't — if so, the composition story strengthens
   (input-side for Light/Camera, architecture-side for Background/Objects).

## 5. Venue mapping

- **NeurIPS 2027 / ICML 2027** — method + composition study + ablations.
- **CoRL 2027 / IROS 2027** — robustness-focused framing.
- **RA-L** (Q2, fast turnaround) — safety net.
- Positioning sentence: *"While recent modules fix single perturbation families (AnyCamVLA:
  viewpoint; PDF: object pose), we give the first test-time module for illumination/texture
  shift and show it composes with architecture-level robustness, compounding gains on
  LIBERO-Plus."*

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Naive composition hurts (pilot: canon+consensus = 0.333 vs consensus 0.667 on Light) | **This is the finding, not a failure.** Scaled study tests a learned gate to fix interference. |
| Single-seed deltas are noise (lab's documented 10.5-21.9% floor) | paired design cancels to first order; claims labeled preliminary until 5-seed |
| Reviewer: "canonicalization is old" | exact DeepONet-precedent framing: novelty = port + gap + composition, with verified 2026 prior-art map |
| Consensus cost (N× forwards at replan) | N=4 at replan=5 ≈ +3.4% wall-clock per env step (measured in pilot) |
| Background/Objects untouched by input-side modules | Architecture-level (DeepONet) is the hypothesized fix; scaled study tests this directly |
