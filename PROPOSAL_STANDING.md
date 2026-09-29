# Where the proposal actually stands — method-by-method

Compiled from the sealed `_confirm` results (k=4 primary, k=2 secondary; n=400/task except
PickCube n=293, pre-registered). All success rates are live closed-loop, DP policy, 8 ManiSkill
tasks. Stats are the pre-registered exact McNemar + Holm from `analyze_confirm.py` — nothing here
is re-derived by hand.

## 1. The method family (this is the "more than one method" confusion)

You built **one core idea — a small learned MLP that reconstructs the within-chunk action shape —**
and evaluated it under four decoding heads. They are not four independent methods; they are one
method (learned chunk-shape reconstruction) plus its ablations and one standalone control:

| Arm | What it is | Role |
|---|---|---|
| `qp_learned` | learned shape anchor + **QP governor** (exact block sums, box) | **the flagship method** |
| `learned_raw` | learned anchor, block sums restored, clipped (no QP) | ablation: is the QP needed? |
| `learned_tanh` | same MLP → whole chunk through tanh (bounded, no QP) | ablation: "why not just bound the net?" |
| `mlp_bc` | same MLP as the whole policy (no DP, **runs at native rate**) | control: "why not just the small net?" |

Baselines in the same runs: `zoh`, `tac_fold_satfix`, `qp_anchor` (ARC, non-learned),
and `native` (rate ceiling — no reduction).

## 2. Success rates (mean over 8 tasks)

| Method | k=4 | k=2 | Reading |
|---|---:|---:|---|
| `native` (ceiling, no reduction) | 81.8 | 81.8 | the target, not a competitor |
| `mlp_bc` (small net, **native rate**) | 72.9 | 72.9 | rate-free — not an apples-to-apples 4X method |
| `learned_tanh` | 60.0 | 75.6 | simpler decoder, matches/beats flagship |
| **`qp_learned` (flagship)** | **59.2** | **78.7** | best rate-reduced method at k=2 |
| `learned_raw` | 57.4 | 77.2 | ablation |
| `qp_anchor` (ARC, non-learned) | 36.7 | 71.8 | best post-hoc learned-free |
| `tac_fold_satfix` | 34.3 | 71.3 | prior post-hoc |
| `zoh` | 30.9 | 66.8 | naive |

## 3. The pre-registered verdicts — good vs bad

### GOOD (real, citable, pre-registered wins)

- **H1 (primary) — PASS.** `qp_learned` beats the best post-hoc resampler (`tac_fold_satfix`)
  with **5 Holm-significant wins, 0 losses** (PickCube, RollBall, PullCube, AnymalC, StackCube).
  This clears the exact rule you sealed (">=4/8 wins, 0 losses"). **This is your headline result.**
- **H2 — PASS (strong).** `qp_learned` beats the non-learned `qp_anchor` on the same 5 tasks.
  **Learning the action shape matters vs a fixed anchor.** This is the real scientific finding.
- **The framing "control-rate conversion is a hidden failure source" is strongly supported:**
  at k=4 the post-hoc methods collapse (zoh 31%, tac_fold 34%) while the learned family recovers
  to ~59–60%. That gap is the story.

### BAD / WEAK (the honest problems)

- **H6 — FAIL.** `learned_tanh` (the simplest bounded decoder) is numerically as good or better
  on 6/8 tasks. **The QP governor has no demonstrated success-rate advantage over a plain tanh net.**
- **H4 — NULL.** vs `learned_raw`, nothing survives Holm. **The QP box/conservation projection
  itself is not what's buying the gain.**
- **H3 — DAMAGING to a "governor is the reason" claim, but read the caveat.** The standalone
  `mlp_bc` beats `qp_learned` significantly on 4 tasks (LiftPeg −82.7pp, PushCube −31.8pp,
  PickCube −25.6pp, PokeCube −7.0pp). **Caveat: `mlp_bc` is rate-free** (it never takes `K`,
  runs at native rate — verified in `fit_bc`/`decode_bc`), so it is not a fair 4X rate-conversion
  competitor. But per your own pre-reg wording, on most tasks the small net alone is competitive,
  so the paper **must not** claim the governor is the active ingredient.

## 4. The bottom line — is the proposal good or bad?

**It is a solid method paper core with one over-claimed mechanism. Split it:**

- **What is genuinely good and defensible:** *learned action-shape reconstruction beats post-hoc
  resampling for control-rate conversion of frozen policies* (H1 + H2, both pre-registered, both
  pass). That is a real, citable contribution. Reframe the proposal around **this**, not around the
  QP governor.
- **What is not supported and should be dropped or demoted:** the claim that the **QP conservation
  governor** is the reason it works (H4 null, H6 fail) and that it beats a small standalone net
  (H3 mixed/damaging). Your simpler `learned_tanh` decoder matches the flagship — that is a *good*
  problem (it means the method is simpler than you thought), but it kills the QP-mechanism story.

**Recommended claim (safe):** *"A learned chunk-shape reference, decoded simply, recovers most of
the loss from control-rate reduction and beats every post-hoc resampler we tested (pre-registered,
5/8 Holm wins, 0 losses). The gain comes from learning the shape, not from the QP projection."*

**Your dossier's own rating is accurate:** ~4/10 as-is, ~6–7/10 after the open gates (external
spline/BSP baseline, more than one training seed, narrow claim wording). I agree with it.

## 5. The two things that most improve your standing

1. **Training-seed robustness (currently Weak).** One checkpoint per task. Run 2–3 seeds on the
   5 H1-win tasks and show the win holds — this is the single biggest credibility upgrade.
2. **One external baseline (currently Missing).** A same-budget B-Spline Policy / Spline Policy
   reproduction on 1–2 tasks. If `qp_learned` stays ahead, the narrow claim hardens; if it doesn't,
   you still have a diagnostic paper.
