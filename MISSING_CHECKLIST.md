# Everything missing — ARC paper(s) + thesis (compiled 2026-09-29)

[x] = already exists / done. [ ] = missing. Ordered by how much it matters.

## A. Evidence gaps (ARC vs spline/B-spline) — THE core gap
- [ ] **6/8 ManiSkill tasks never paired vs spline/B-spline**: AnymalC-Reach, LiftPegUpright,
      PokeCube, PushCube, RollBall, StackCube. ARC ran there but never against the baselines on the
      same episodes. Run: `harness.py TASK --k 4 --arms native,zoh,spline,spline_satfix,bspline_eps_raw,qp_anchor`
      at each task's confirm window. Cheapest, highest-value fix.
- [ ] **Training-seed robustness**: every task = 1 checkpoint. Need 2-3 DP seeds on the win tasks
      (PickCube, PullCube, PushT-k3 minimum).
- [ ] **Pre-registered PushT P1 FAILED** (qp_anchor lost to raw spline at k=2; 8.4 vs 9.1 suite avg).
      Must be reported as a loss; blocks all "beats spline everywhere" wording.
- [ ] Root-cause the PushT loss: is PushT saturation-free so the box constraint never engages?
      Check policy_raw_sat_frac; write the finding up as analysis.
- [x] PickCube k=4 paired face-offs (splinefaceoff n=293; n400_anchor n=400).
- [x] PullCube k=4 paired; RoboCasa n=100 paired cells (but see void-rule item in B).

## B. Statistical rigor (house-rule violations to fix NOW)
- [ ] **Holm correction over the ARC-vs-spline/B-spline family.** assemble_arc_vs_spline.py reports
      3 raw p<0.05 "WIN" cells out of 28 comparisons, NO correction. House rules require Holm across
      the family. PickCube/PullCube likely survive; PushT-k3 (p~0.01-0.02) likely will not.
- [ ] **Cluster-aware aggregate**: pooled episode-level McNemar (p=1e-6) is anti-conservative
      (episodes within a task correlate). Use per-task meta-analysis or drop the pooled claim.
- [ ] **Pre-register the ARC-vs-spline comparison itself** (existing preregs cover PushT and
      qp-vs-tac_fold only; none defines "primary endpoint = mean SR over suites").
- [ ] Power analysis for +2pp margins: what n resolves it? (decides if "best average" is more than
      a ranking).
- [ ] Void/floor-ceiling rule stated and applied consistently (RoboMimic 80-100%, RoboCasa 6-36%).
- [x] Exact McNemar, paired episodes, checkpoint sha256 discipline.

## C. Method / ablation gaps
- [ ] **Anchor ablation**: qp_anchor anchors on TAC-Fold (`anchor_fn or resample_tac_fold`). Test
      anchor = linear/ZOH/spline/zero (=resample_qp). Unanswered reviewer question; also the only
      remaining trace of tac_fold in the story, so it must be justified or replaced.
- [ ] **QP runtime table**: SLSQP ms/chunk at k=2/4/8 (partial data in batched_timing log).
- [ ] **Reconstruction-error curves vs k** from the n=993 open-loop replay (RESULTS_QP_DRAFT Table 1
      -> figures + error bars). This is the SPL letter's hero experiment.
- [ ] 1D synthetic demo (box-constrained recon vs cubic/PCHIP overshoot) — 1 figure, makes the
      method self-contained without robots.
- [ ] Parameter-sensitivity statement: QP is parameter-free (no tunable weight) — state + verify.
- [x] Feasibility + exact block-sums provable by construction.

## D. SPL letter (IEEE Signal Processing Letters) — what it needs
- [ ] Reframe title/abstract: "box-constrained multirate signal reconstruction", NOT robot policy.
- [ ] One proposition: feasibility + exact block-sum interpolation (short proof).
- [ ] Cite the DSP literature: multirate/SRC (Crochiere-Rabiner, Farrow), constrained interpolation,
      saturation-limited signals. Currently cites robot papers only.
- [ ] Metrics as reconstruction methods: recon MSE + overshoot rate + block-sum violation; robots
      demoted to a 1-paragraph application.
- [ ] Method property matrix vs ZOH/spline/PCHIP/TAC-Fold/B-spline (exact block sums? box-feasible?
      cross-block? causal? parameter-free?).
- [ ] 4-5 page format; drop all benchmark-breadth claims.

## E. JBE paper (Journal of Broadcast Engineering) — what it needs
- [ ] Frame as bandwidth-constrained control-signal transmission / teleoperation over rate-limited
      links (their Transmission System + Media Service topics), NOT robot learning.
- [ ] Bandwidth/latency table: bits-per-command + effective control rate at k=1/2/4/8.
- [ ] Cite the scope precedent: their Vol 31(4) includes sonar time-frequency + UAV tracking papers,
      so applied SP is arguable; include 1-2 of their own applied-SP papers as positioning.
- [ ] Must NOT be a reworded SPL letter (self-plagiarism) — different hero experiment (transmission
      metrics), different framing.

## F. Thesis / defense gaps (the size-of-idea problem)
- [ ] The thesis-sized principle: promote the saturation crossover (fig_crossover_pickcube.png) from
      a figure to a predictive claim ("rate reduction breaks a policy iff per-block commanded motion
      exceeds the box" + measurable threshold).
- [ ] Unifying framework: ZOH/spline/TAC-Fold/satfix/QP as special cases of one constrained
      reconstruction objective. Turns "our method wins" into theory.
- [ ] VLA-scale evidence: everything is state-based DP (+ limited FM). No vision-language backbone,
      no language, no real robot. Dossier's own words: "does not establish a VLA result".
- [ ] External learned baselines (matched budget): Spline Policy / B-Spline Policy / CAT — entirely
      missing; the learned versions of your baselines are untested.
- [ ] Second policy family at scale (FM exists only for PickCube k=4 n=100).
- [ ] Decide the fate of the rate-conditioning direction (head=arc, retired from ARC) — own thesis
      chapter or cut.

## G. Documentation / consistency (cheap, do this week)
- [ ] **PROPOSAL_STANDING.md is STALE** (still "one MLP + four heads", predates ARC=qp_anchor).
      Rewrite to the lineage framing: tac_fold -> qp -> ARC(=qp_anchor); learned family = separate
      question; bar = best avg SR + suite-level wins.
- [ ] Naming sweep: grep PAPER_DRAFT_SCOPED.md, Q2_PUBLICATION_EVIDENCE_DOSSIER.md, JEV_VLA_MODEL.md,
      RESULTS_*.md for "ARC" meaning resample_arc or the rate-conditioned head; fix.
- [ ] Close .remember/ false alarms: mlp_bc identical k=2/k=4 (by design, rate-free arm) and
      PickCube n=293 (pre-registered window) are still listed as open issues.
- [x] index.html (multi-rate/full_grid_2026-09-07/) RESOLVED 2026-09-29: the uncommitted hand-edit
      claimed a fabricated "8-task, n=3,093 paired" spline comparison (71.76 vs 69.35/66.94 — those
      spline numbers exist in NO result file; the 3,093 episodes are the confirm runs which have no
      spline arms). Reverted (diff preserved in index.html.handedit.patch). index.html is now the
      same generated honest report as arc_benchmark_report.html (written by assemble_arc_vs_spline.py).
      NOTE: a parallel session had also slipped a 755-line inflated variant into commit 3c97122
      (71.76/97.33 inside) — the generator output now supersedes it.
- [ ] Align every paper claim with RESULTS_QP_ANCHOR_CITABLE.md's scope (PickCube-only + the PushT-k2
      exception) before submission.
- [x] arc_benchmark_report.html regenerated from raw JSONs; resample_arc retired; legacy table
      builders deprecated.

## Top 5 if you only do five things
1. Run the 6 missing ManiSkill paired face-offs (A) — turns "wins 2 tasks" into "wins the suite".
2. Holm-correct the ARC-vs-spline family and re-derive which wins survive (B) — house rule, blocks
   submission until done.
3. 2-3 training seeds on the win tasks (A).
4. Rewrite PROPOSAL_STANDING.md + naming sweep (G) — one hour, removes all remaining confusion.
5. Write the SPL letter skeleton with recon-error framing (D) — ~70% of data already exists.
