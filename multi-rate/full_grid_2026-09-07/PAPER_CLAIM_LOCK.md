# Paper claim lock — 2026-09-23

## Single contribution

**A plant-free, constraint-preserving governor for changing the execution rate of frozen action-chunk robot policies.**

The governor reconstructs a higher-rate action stream while enforcing two invariants at decode time:

1. every executed action stays inside the actuator box; and
2. every coarse block preserves the policy's commanded aggregate displacement.

The reference-anchored objective selects, among feasible reconstructions, the stream closest to a chosen action-shape reference. The contribution is the constraint set, its projection/governor formulation, and the evidence that these invariants matter during control-rate conversion. It is **not** "using a QP," smoothing actions, or claiming universal performance gains.

## Claims allowed now

- Existing VLA/action-chunk evaluation can hide failures caused by changing the executed control rate.
- Unconstrained interpolation and clipping can violate the policy's commanded block displacement.
- The implemented governor enforces the box and block-sum constraints by construction.
- Current results support task-conditional gains, not universal superiority.

## Claims blocked until their gates pass

- **"First" or novel:** blocked until the dated literature search and closest-method comparison are complete.
- **Better than the strongest external method:** blocked until a same-task, same-policy, same-rollout external baseline is run.
- **General improvement:** blocked until the pre-registered multi-task confirmation succeeds without hidden task selection.
- **Q1-ready:** blocked until both the external-baseline and second-task confirmation gates pass and every cited table is regenerated from audited raw JSON.
- **Real-world relevance:** blocked until a real-policy or real-robot cell exists; simulation breadth alone is not a substitute.

## Excluded from the main contribution

REST/input restoration, DeepONet, GRPO, ARC, the learned-shape governor, and gripper synchronization are separate candidate projects or ablations. They must not appear as additional headline contributions in this paper. The learned-shape governor may enter only as an ablation or extension after its sealed confirmation finishes and passes its own pre-registered rule.

## Evidence hierarchy

1. Primary: paired closed-loop success under changed execution rate.
2. Supporting: paired open-loop constraint violations and reconstruction error.
3. Mechanism: feasibility, conservation, identity-on-feasible-input, and solver correctness checks.
4. Context only: pilot runs, development windows, uncorrected comparisons, and internal variants.

## Frozen decision rule

If the external baseline and fair multi-task/second-task confirmation both succeed, target a Q1 robotics venue. If either fails, retain the exact same scoped paper and target RA-L/Q2 or a workshop; do not add unrelated modules to manufacture breadth.
