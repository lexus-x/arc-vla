# Scoped positive method paper — draft (honest claim set)

Status: DRAFT 2026-09-28 · numbers machine-verified by `report/rebuild_eval_table.py` where
marked [A]; [B] audited docs; [C] legacy cells pending manifest (see PAPER_AUDIT_STATUS.md).

## Working title

**Constraint-Preserving Action-Stream Governors Beat Interpolation on Stiff-Contact
Manipulation under Control-Rate Conversion**

## Abstract

Frozen action-chunk robot policies are routinely evaluated at the control rate they were trained
on, yet deployment frequently requires executing their action stream faster or slower than that
native rate. We show that this control-rate conversion is a large, hidden failure source: on
PickCube, executing the same frozen Diffusion Policy at k=4 drops success from 88.0% to 40.5%
under zero-order hold [A]. We introduce a family of plant-free, training-free governors —
TAC-Fold+satfix and QP-Anchor — that reconstruct the action stream subject to two invariants
enforced at decode time: every executed action stays inside the actuator box, and every coarse
block preserves the policy's commanded aggregate displacement. On PickCube at k=4 (n=400 paired
closed-loop episodes) QP-Anchor beats cubic spline + satfix (+3.2 pp, exact McNemar p=0.0106),
TAC-Fold + satfix (+3.2 pp, p=0.0106) and B-Spline + satfix (+13.0 pp, p=5.3e-9), all
Holm-significant [A]. Across eight stiff-contact ManiSkill tasks at 2x slower the governor family
takes the top two slots ahead of both interpolants [A/C]. On smooth human-demonstration suites
(LIBERO, RoboMimic, RoboCasa) cubic spline remains the best interpolant on average, and we
report this in full: the contribution is regime-scoped superiority plus deterministic
conservation guarantees, not universal dominance.

## Claims table (the paper's entire claim set)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Control-rate conversion of frozen policies is a hidden failure source | ZOH collapse: 88.0%->40.5% (PickCube k=4, n=400) [A]; ZOH < 65% on 4/8 ManiSkill tasks at k=2 [A] | **Supported** |
| 2 | QP-Anchor beats cubic spline, TAC-Fold and B-Spline significantly on PickCube k=4 | +3.2/+3.2/+13.0 pp, p=0.0106/0.0106/5.3e-9, Holm m=3, split-half consistent [A + RESULTS_QP_ANCHOR_CITABLE.md] | **Supported** |
| 3 | Governor family beats both interpolants on the stiff-contact category (8 ManiSkill tasks, k=2) | QP-Anchor 71.75%, TAC-Fold+sf 71.29% vs Spline 70.79%, B-Spline 70.00% [C interp cells] | **Point estimates; paired spline rerun needed** |
| 4 | Constraint enforcement recovers the ZOH collapse on every hard task | +4.8..+11.0 pp over ZOH on all 4 tasks with ZOH<65%, max p=0.0344 [A] | **Supported** |
| 5 | Box feasibility + block conservation hold by construction (identity on feasible input) | solver correctness + feasibility checks [B mechanism] | **Supported (provable)** |
| 6 | Open-loop reconstruction dominates the interpolants | 7/9 Holm-significant wins, n=993 paired replay [B] | **Supported (audited)** |
| 7 | Extension: learned shape governor beats TAC-Fold at k=4 | 5/8 Holm-significant, no significant loss [B, Q2 dossier] | **Supported, separate pre-reg family** |

Explicitly **not claimed**: universal superiority over spline; VLA-backbone results (confirmation
runs are state-based Diffusion Policy); real-robot relevance; training-seed robustness (one
checkpoint per task); "first" (literature gate open).

## Results narrative (for the results section)

**R1 — The failure being fixed.** Same frozen policy, same initial states, only the executed
control rate changes. PickCube k=4: native 88.0% vs ZOH 40.5% vs B-Spline+satfix 40.5% [A].
Half the native performance disappears before any method choice is made; the paper's job is to
recover it with guarantees.

**R2 — Headline closed-loop win (fully paired, Holm-corrected).** PickCube-v1, k=4, n=400,
Diffusion Policy. QP-Anchor 53.5% (214/400) vs cubic spline+satfix 50.2% (201/400, discordant
18/5, p=0.0106), vs TAC-Fold+satfix 50.2% (p=0.0106), vs B-Spline+satfix 40.5% (discordant 67/15,
p=5.3e-9). Holm m=3: all three significant. Split-half direction-consistent; second policy family
(Flow Matching, n=100) direction-consistent. Source: `result_dp_PickCube-v1_k4_n400_anchor.json`
(SHA-256 recorded in report) [A].

**R3 — Contact-category sweep (k=2, 8 ManiSkill tasks).** Task-weighted means: QP-Anchor 71.75%,
TAC-Fold+sf 71.29%, Spline 70.79%, B-Spline 70.00%. Both governor arms beat both interpolants on
the category. Our cells cross-check exactly against the 8 sealed confirm JSONs (n=293/400) [A];
interpolation cells are legacy estimates awaiting paired reruns [C]. Per-task detail: our margin
concentrates on LiftPeg (+2.8..+9.9), AnymalC (+1.5), RollBall (+1.5), StackCube (+0.7..+1.5);
PickCube spline leads there (87.0 vs 83.6) — disclose per-task table.

**R4 — Where constraints matter: vs ZOH.** On all tasks where ZOH falls below 65% (LiftPeg,
RollBall, StackCube, AnymalC) both governors recover +4.8..+11.0 pp (max p=0.0344) [A]; on ceiling
tasks (PullCube, PokeCube, PushCube) all arms tie within noise. Saturation fraction is the effect
modifier (`policy_raw_sat_frac` in the confirm JSONs) — mechanism-consistent.

**R5 — Guarantees.** By construction: |v|<=1 and exact per-block conservation, identity on
feasible input, sub-second/episode SLSQP, zero learned parameters. These hold regardless of task
success; they are the deployability argument when success deltas are ties.

**R6 — Honest counterweight (goes in the paper, not the drawer).** Grand average over 19 tasks:
spline 70.16% combined mean is the best single method; spline recovers 99.3% of native at 2x
faster on smooth suites. PushT reversals favor interpolants. The paper's framing: interpolation
is the right default for smooth, saturation-free streams; governors are the right default when
actuator limits and displacement conservation bind.

## Limitations (verbatim-safe)

One training seed per task; episode p-values are not seed robustness. State-based DP for
confirmation runs — call it a "robot policy", not a VLA. Simulation only. Interpolation cells in
R3 lack paired booleans locally. PushT has two conflicting sample sizes (n=100 legacy vs n=400
audited) — report the n=400 paired one as primary. MT50 N=6/env is weak evidence.

## Gates before submission (from PAPER_CLAIM_LOCK / Q2 dossier)

1. **Paired interpolation reruns** on the 8 ManiSkill k=2 tasks (or restrict claim 2 to
   PickCube/claim 3 to point estimates).
2. **Manifest-verify every [C] cell** (source JSON, SHA-256, protocol, n) — the report marks them.
3. **3 training seeds** for the primary method on primary tasks (Gate B).
4. **Matched learned-spline baseline** if the learned governor enters the headline (Gate A);
   otherwise keep it as the labeled extension.
5. Claim language per Gate C: "regime-scoped gains + deterministic guarantees", never "generally
   better".

## Why this is a positive paper

It wins exactly where its mechanism predicts (saturation/contact), with Holm-significant paired
evidence on the flagship task, plus provable invariants — and it openly scopes away the suites
where spline wins. Reviewers reject hidden losses, not scoped wins. Venue: RA-L / Q2 with gates
1-3 closed; workshop tier without them. Per the frozen decision rule in PAPER_CLAIM_LOCK: do not
add unrelated modules to manufacture breadth.
