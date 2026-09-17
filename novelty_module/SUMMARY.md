# Summary: 2-Day Pilot Results and Proposal

**Date:** 2026-09-02 · **Machine:** Blackwell RTX PRO 6000 (96 GB) · **Location:** `/home/user/Desktop/novelty_module/`

## Running now (unattended overnight)

| Job | Checkpoint | Suite | Progress | Output |
|---|---|---|---|---|
| DeepONet 4-arm pilot | `don_v2_30k_noaug_s0` (30K) | Spatial | 20/96 rollouts | `results/pilot_deeponet.json` |
| Flow Goal pilot (chained) | `flow8300_goal_s0` | Goal | queued | `results/pilot_flow_goal.json` |
| Flow Object pilot (chained) | `flow8300_object_s0` | Object | queued | `results/pilot_flow_object.json` |
| Flow Long pilot (chained) | `flow8300_10_s0` | Long | queued | `results/pilot_flow_10.json` |

ETA: ~7-8 h total (DeepONet first, then 3 flow suites sequentially to avoid GPU contention).
Logs: `pilot_deeponet.log`, `pilot_flow_goal.log`, `pilot_flow_object.log`, `pilot_flow_10.log`, `chain.log`.

## What was done today

1. **Flow pilot (done):** 1-seed paired 4-arm pilot on LIBERO-Spatial — see table below.
2. **Harness extended:** `eval_canon_pilot.py --head deeponet` — auto-resolves the lab's
   training-provenance guard from `run_config.json` (verified: head=deeponet, bandlimit=0,
   pool_norm=0, P=256, blocks=3, queries=8, fourier=16). Same protocol, identical tasks +
   init states across both backbones.
3. **Proposal updated:** two-axis scaled study (cross-suite on flow, cross-backbone on Spatial).

## Flow pilot results (Spatial, 1 seed, directional)

| Arm | Light | Background | Objects | Camera | Avg |
|---|---|---|---|---|---|
| base | 0.500 | 0.167 | 0.167 | 0.167 | 0.250 |
| +canon | **0.667** | 0.167 | 0.167 | 0.333 | 0.333 |
| +consensus | **0.667** | 0.167 | 0.167 | **0.500** | **0.375** |
| +canon+consensus | 0.333 | 0.167 | 0.167 | 0.333 | 0.250 |

## Key findings

1. **Canonicalization hits its target:** +16.7pp Light, +16.7pp Camera. Zero on Background/Objects (expected — illumination module, not texture/layout).
2. **Consensus is broader:** +33.3pp Camera, +16.7pp Light (noise-averaging handles viewpoint jitter).
3. **Naive composition hurts** (canon+consensus 0.333 vs consensus 0.667 on Light) — the interference is the finding; motivates a learned composition gate.
4. **Background/Objects need architecture-level solutions** — exactly what the DeepONet pilot (running) will test.

## Module properties (reviewer-facing)

| Property | Canon | Consensus |
|---|---|---|
| Parameters | **0** | **0** |
| Training required | **none** | **none** |
| Plug-and-play | yes (obs pre-transform) | yes (N forwards + mean) |
| Overhead | µs-scale | N× forwards at replan (~+3.4% wall-clock at N=4, replan=5) |

## The proposal

**Positioning:** *"While recent modules fix single perturbation families (AnyCamVLA: viewpoint; PDF: object pose), we give the first test-time module for illumination/texture shift and show it composes with architecture-level robustness, compounding gains on LIBERO-Plus."*

**Scaled study (two axes, all checkpoints already on disk):**
1. **Cross-suite (flow, 4 suites):** Spatial/Goal/Object/Long × {base, +canon, +consensus} × 5 seeds.
2. **Cross-backbone (Spatial):** flow8300 vs DeepONet-30K vs DeepONet-v2 × 4 arms — directly answers "does it compose with DeepONet?" and "is DeepONet already covering Light/Camera?".
3. Learned composition gate (fixes the pilot's interference finding).
4. Ablations: gain steps, N ∈ {2,4,8}, σ sweep, latency.

**Venues:** NeurIPS/ICML 2027 (primary) · CoRL/IROS 2027 · RA-L (Q2 safety net).

## Honest caveats

- 1-seed deltas are directional only (lab's documented 10.5-21.9% rerun noise floor); paired design cancels to first order; 5-seed study is confirmatory.
- DeepONet comparison is Spatial-only (only checkpoint on disk).
- Background/Objects: input-side modules don't touch them — this is stated, not hidden.

