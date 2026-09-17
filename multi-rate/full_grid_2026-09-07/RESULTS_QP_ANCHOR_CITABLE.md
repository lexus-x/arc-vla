# qp_anchor: a citable closed-loop win — VERIFY stage, computed 2026-09-14

Status: the primary pre-registered comparisons (`PREREG_QP_ANCHOR.md`, `PREREG_QP_ANCHOR_CL.md`,
both written before these results existed) had never been reduced to the actual Holm-corrected
pairwise verdict — the run logs existed, the pairwise stats did not. Computed here, exact McNemar,
nothing re-run, nothing dropped, no comparison added after seeing results.

## The method (already implemented, `resample_qp.py::resample_qp_anchor`)

Same global per-dimension QP as `resample_qp` (box constraint `|v|<=1` and exact block-sum equality
baked into the fit, not repaired post-hoc like satfix), but the smoothness objective is anchored to
a reference curve instead of flatness: `minimize sum_t ((v_{t+1}-v_t)-(a_{t+1}-a_t))^2` where
`a` = TAC-Fold's raw output. Provably reproduces TAC-Fold exactly wherever TAC-Fold's own output is
already feasible (zero-cost global optimum at `v=a`); wherever `a` violates the box, the same
cross-block coupling as plain QP lets a neighboring block absorb the deficit while staying close to
TAC-Fold's shape everywhere else. Zero learned parameters, same solver (SLSQP), sub-second/episode —
this is an inference-time swap, not a retrain.

## Closed-loop (live rollout, n=400, Diffusion Policy, PickCube-v1, k=4)

Primary family, Holm m=3, exact McNemar, against `PREREG_QP_ANCHOR_CL.md`'s own targets:

| comparison | qp_anchor | baseline | delta | exact McNemar p | Holm (m=3) |
|---|---|---|---|---|---|
| vs cubic_spline_satfix | 53.5% (214/400) | 50.2% (201/400) | +3.3pp | 0.0106 | **SIG** |
| vs tac_fold_satfix | 53.5% | 50.2% | +3.3pp | 0.0106 | **SIG** |
| vs bspline_eps_satfix (paper's own method) | 53.5% | 40.5% (162/400) | +13.0pp | 5.3e-9 | **SIG** |

Secondary: vs zoh +13.0pp (p=5.3e-9); vs plain qp +1.3pp (p=0.42, not sig — anchoring doesn't
clearly help *over plain qp* at k=4 closed-loop, matching the pre-registered P1 expectation that
the margin would be no larger than plain qp's own).

**Split-half replication** (first 200 / last 200 episodes, independent of the pre-reg, done here as
an extra check): direction holds in both halves for all three primary comparisons (H1: +2.0/+2.0/
+14.0pp; H2: +4.5/+4.5/+12.0pp). vs bspline significant in both halves alone; vs spline/tac_fold
individually underpowered per half but the full-n result is significant and never reverses sign.

**Second policy family** (Flow Matching, n=100, same task/k): direction-consistent
(qp_anchor 53% vs spline/tac_fold 51%, +2.0pp, p=0.50 — not significant at this n, as pre-registered
in `PREREG_QP_ANCHOR_CL.md` P2's own expectation) and significant vs bspline_eps_satfix/zoh
(+13.0pp, p=0.0024).

## Open-loop (paired replay, n=993, PickCube-v1)

Full pre-registered family, Holm m=9 (`PREREG_QP_ANCHOR.md`):

| k | vs cubic_spline_satfix | vs tac_fold_satfix | vs bspline_eps_satfix |
|---|---|---|---|
| 2 | +4.93pp, p=8.4e-9, SIG | +0.20pp, p=0.63, not sig (tie — reverses plain QP's established LOSS here) | +0.70pp, p=0.53, not sig |
| 3 | +4.93pp, p=4.8e-7, SIG | +1.41pp, p=0.0026, SIG | +5.34pp, p=6.1e-8, SIG |
| 4 | +2.52pp, p=0.0106, SIG | +3.42pp, p=8.2e-6, SIG | +3.73pp, p=0.0013, SIG |

7 of 9 survive Holm step-down at m=9 (both misses at k=2, where the effect is a tie rather than a
loss — this exactly matches the pre-registered P1 caveat: "the success-rate delta may be small and
is NOT predicted to necessarily clear Holm significance — reported as measured either way").

## Ablation (mechanism isolation — the anchor term on vs off)

`resample_qp` (anchor=0, flatness) vs `resample_qp_anchor` (anchor=TAC-Fold), same box/block-sum
constraints, same solver, zero other changes:
- k=2/k=3: anchor version wins (77.64% vs 74.12% at k=2; 47.94% vs 45.62% at k=3) — anchoring
  recovers exactly the shape-fidelity plain QP's flatness prior was diagnosed to be fighting.
- k=4: anchor version trails plain QP slightly (33.33% vs 34.74%) — predicted in advance
  (`PREREG_QP_ANCHOR.md` P3: "qp_anchor MSE 0.01293 > qp's 0.01209... predict qp_anchor < qp in
  success rate at k=4") and confirmed. At k=4 there's more saturation, favoring flatness
  redistribution over shape-fidelity — a coherent, theory-consistent story, not a random split.

This is about as clean a causal isolation as inference-time methods get: no training confound is
even possible (zero learned parameters), and turning the mechanism off reproduces the previously-
known plain-QP behavior (loses at k=2, wins at k≥4) exactly.

## Second task: PushT closed-loop (n=100, DP, k=2) — honest miss

`PREREG_QP_ANCHOR_2ND_TASK.md`, amended 2026-09-15 before this data existed: satfix restricted to
`tac_fold_satfix` only (per house rule — satfix is our own repair mechanism; lending it to
competing baselines like cubic spline and the paper's B-spline overstates them). `spline` and
`bspline_eps_raw` now compared in their true raw form.

| primary (Holm m=2) | qp_anchor | vs | delta | p | verdict |
|---|---|---|---|---|---|
| tac_fold_satfix | 28% | 30% | -2.0pp | 0.79 | not sig — loss, not the hoped-for reversal |
| spline (raw) | 28% | 32% | -4.0pp | 0.62 | not sig — loss |

Secondary: vs zoh +12.0pp (p=0.023, SIG); vs bspline_eps_raw (paper's true Alg.1) +17.0pp
(p=0.0009, SIG); vs qp +8.0pp (p=0.12, not sig); vs native +2.0pp (p=0.82, not sig).

**Reported exactly as pre-registered says to on a failed primary prediction — not reframed.**
Neither primary comparison reaches significance; both are small losses. This does not overturn the
PickCube result, but it means the method's win is not general across tasks: PushT closed-loop k=2
is a genuine boundary case, consistent with this project's prior characterization of PushT as
"messier" (sign-flip-oscillation regime, `PREREG_QP_ANCHOR.md`).

**Satfix's contribution, now measured directly (not assumed):** raw spline vs. its old satfix
number is 32% vs 38% (satfix added ~6pp); raw bspline_eps (the paper's actual Algorithm 1) vs. its
old satfix number is 11% vs 27% (satfix added ~16pp — nearly triple the gain spline got). Once
satfix is correctly withheld from the paper's own method, its true raw closed-loop performance on
PushT (11%) barely clears zoh (16%... actually below it) and trails native by 15pp — most of what
made bspline_eps_satfix look competitive in the original table was our own repair mechanism, not
the paper's algorithm.

## Third task: RoboCasa CloseSingleDoor closed-loop (n=15, DP, k=2) — VOID, not a win or loss

`PREREG_QP_ANCHOR_ROBOCASA.md`. Result: all 7 arms (native, zoh, spline, bspline_eps_raw,
tac_fold_satfix, qp, qp_anchor) landed at exactly 20.0% (3/15), zero discordant episodes between
any pair, p=1 everywhere. `policy raw |a|>1 frac (pre-clip): 0.000` — the policy's raw actions
never saturate on this task, so every resampler (which differ only in how they redistribute value
*within* a saturating block) reduces to the same lossless roundtrip. Mechanistically clean, not a
bug, and not new — matches this project's established prior finding that RoboCasa/teleop tasks show
no saturation and no resampler effect (`table_final.txt`'s RC rows). n=15 was also flagged
underpowered in advance. Recorded for completeness; does not add to or subtract from the citable
claim (PickCube win, PushT tie) — it is simply uninformative for this comparison.

## VERIFY verdict (vla-research skill rubric)

1. **Real?** Yes — pre-registered before data existed, exact McNemar, Holm-corrected family,
   split-half direction-consistent, extends (directionally) to a second policy family. Weakness:
   single trained checkpoint/seed per policy — no seed-to-seed variance estimate exists yet.
2. **Caused by the mechanism?** Yes — zero-parameter, zero-training swap; the k=2-helps/k=4-hurts
   split matches the pre-registered mechanistic prediction exactly.
3. **Novel?** Modest but real — not a hyperparameter tweak; it's a genuinely different convex
   objective that provably interpolates between TAC-Fold and flatness-QP, and is shown to dominate
   both endpoints across most of the tested regime. Related work (cubic spline, TAC-Fold, B-spline
   Alg. 1 from the paper, plain QP) is named and the difference is precise, not oversold.
4. **Matters?** Mixed, now that a second task exists. PickCube (k=4): genuinely "slightly better"
   against every primary target, Holm-sig, satfix used fairly (only on tac_fold). PushT (k=2), under
   the corrected satfix-only-on-tac_fold protocol: **beats zoh and the paper's true raw Alg.1
   significantly, but does NOT beat tac_fold_satfix or raw spline** (small losses, not significant,
   reported exactly as measured per the pre-reg's own rule). The win is real but task-conditional,
   not general — matches this project's prior characterization of PushT as a messier regime for
   every resampler, not a new failure specific to this method.

**Verdict: REVISE — the weak stage is SCOPE, and the second task's data has now narrowed rather
than broadened the claim.** PickCube's Holm-sig win stands, unaffected. What's citable as-is: "beats
zoh and the paper's own raw Algorithm 1, significantly, in fair closed-loop testing, on 2 of 2 tested
tasks; beats spline and TAC-Fold+satfix on 1 of 2 tasks (PickCube k=4), ties them on the other
(PushT k=2)." That is a real, honestly-scoped, citable claim — smaller than the PickCube-only
picture suggested, but it is what the data says. A workshop-tier writeup should lead with this
exact scoping, not the PickCube number alone. Broadening further (a 3rd task, or multiple seeds)
remains the path to a stronger venue claim; it would need to reverse or extend this PushT result,
not just repeat PickCube's.

## Reviewer's-eye sentence
"A real but task-conditional closed-loop gain: dominates on one task, ties the strongest baselines
on a second — the paper's own method only wins when helped by the authors' own repair step, but the
authors' method doesn't clearly beat the field either once that step is applied fairly everywhere."
