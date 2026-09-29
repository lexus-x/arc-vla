# Q2 / RA-L publication evidence dossier

**Project:** learned shape governor for control-rate conversion of frozen robot policies  
**Evidence cutoff:** 2026-09-25  
**Readiness:** **Needs revision**  
**Bottom line:** the preregistered superiority claim against TAC-Fold passed, but the evidence does not establish that QP constraint enforcement causes the improvement. A decent Q2 submission is plausible only after a matched learned spline baseline, a narrow claim rewrite, and training-seed replication.

## 1. Decision in one page

### What is already publishable

The following statement is supported:

> On eight preregistered ManiSkill tasks at 4X action resolution, a learned shape reconstructor improved paired closed-loop success over TAC-Fold on five non-void tasks after Holm correction, with no significant loss.

The following statements are **not** supported:

- QP conservation is the reason for the improvement.
- The governor generally beats simpler learned action models.
- The method beats the closest published continuous-trajectory policies.
- The evidence establishes a VLA result; these confirmation runs use state-based Diffusion Policy, not a vision-language-action backbone.
- The method is generally better than native-rate execution.

### Current Q2 survival assessment

| Dimension | Current assessment | Why |
|---|---|---|
| Preregistration and reporting | Strong | Protocol was sealed before confirmation; losses, nulls and void cells remain visible. |
| Episode-level statistical evidence | Strong | 3,093 paired evaluation episodes per arm across eight tasks; exact McNemar tests and Holm families. |
| Training-seed robustness | Weak | One trained policy/reconstructor seed per task. Episode count does not estimate checkpoint-to-checkpoint variance. |
| Causal mechanism | Weak | QP-learned did not beat learned-tanh; standalone MLP dominated on four tasks. |
| External baseline | Missing | No same-task, same-checkpoint-budget reproduction of CAT, Spline Policy or B-Spline Policy. |
| Benchmark breadth | Moderate | Eight tasks, but one simulator family, state observations, no physical robot and no language. |
| Novelty boundary | Narrow | Continuous, resampleable and frequency-aware action representations are already claimed by CAT, Spline Policy and BSP. |
| Reproducibility | Moderate–strong | Raw paired booleans, checkpoint hashes and source-file hashes exist; legacy table assembly is unsafe and must not be used. |

**Current rating:** approximately **4/10 for a decent Q2 method paper**.  
**After the required gates below:** approximately **6–7/10**, assuming the matched spline baseline is competitive rather than devastating and the manuscript adopts the narrow claim.

## 2. Governing preregistration

Authoritative protocol: `/home/user/Desktop/multi-rate/full_grid_2026-09-07/PREREG_LEARNED_GOVERNOR.md`.

Key locked conditions:

- Eight tasks: PickCube, RollBall, PullCube, LiftPegUpright, PushCube, AnymalC-Reach, PokeCube and StackCube.
- Primary rate: `k=4`.
- Seven tasks use 400 untouched held-out episodes; PickCube uses the remaining 293 untouched episodes.
- Same 30k-step Diffusion Policy checkpoint and 200 training demonstrations per task.
- Eight paired arms evaluated on identical initial states.
- H1 success rule: QP-learned must produce Holm-significant wins over TAC-Fold on at least four of eight tasks, with no Holm-significant loss.
- A comparison is void when both arms are below 10% or both are above 90%.
- No task, arm, checkpoint, window or recipe could be removed after results.

## 3. Complete confirmation results

Values are successes / paired episodes. Percentages are shown in parentheses.

| Task | Native | ZOH | TAC-Fold | QP-anchor | QP-learned | Learned raw | Learned tanh | Standalone MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnymalC-Reach | 365/400 (91.2%) | 5/400 (1.2%) | 25/400 (6.2%) | 50/400 (12.5%) | 336/400 (84.0%) | 346/400 (86.5%) | 358/400 (89.5%) | 338/400 (84.5%) |
| LiftPegUpright | 381/400 (95.2%) | 2/400 (0.5%) | 10/400 (2.5%) | 17/400 (4.2%) | 26/400 (6.5%) | 23/400 (5.8%) | 35/400 (8.8%) | 357/400 (89.2%) |
| PickCube | 249/293 (85.0%) | 110/293 (37.5%) | 135/293 (46.1%) | 138/293 (47.1%) | 178/293 (60.8%) | 174/293 (59.4%) | 181/293 (61.8%) | 253/293 (86.3%) |
| PokeCube | 381/400 (95.2%) | 322/400 (80.5%) | 328/400 (82.0%) | 329/400 (82.2%) | 342/400 (85.5%) | 339/400 (84.8%) | 356/400 (89.0%) | 370/400 (92.5%) |
| PullCube | 386/400 (96.5%) | 186/400 (46.5%) | 205/400 (51.2%) | 232/400 (58.0%) | 387/400 (96.8%) | 372/400 (93.0%) | 390/400 (97.5%) | 390/400 (97.5%) |
| PushCube | 380/400 (95.0%) | 275/400 (68.8%) | 264/400 (66.0%) | 265/400 (66.2%) | 261/400 (65.2%) | 261/400 (65.2%) | 249/400 (62.3%) | 388/400 (97.0%) |
| RollBall | 240/400 (60.0%) | 41/400 (10.2%) | 52/400 (13.0%) | 64/400 (16.0%) | 212/400 (53.0%) | 196/400 (49.0%) | 221/400 (55.2%) | 63/400 (15.8%) |
| StackCube | 144/400 (36.0%) | 9/400 (2.2%) | 30/400 (7.5%) | 30/400 (7.5%) | 86/400 (21.5%) | 62/400 (15.5%) | 65/400 (16.2%) | 80/400 (20.0%) |

## 4. Preregistered hypothesis decisions

### H1: QP-learned versus TAC-Fold — **PASS**

| Task | Difference | Exact p | Holm p | Decision |
|---|---:|---:|---:|---|
| AnymalC-Reach | +77.8 pp | 8.03e-88 | 6.42e-87 | significant win |
| LiftPegUpright | +4.0 pp | 0.0113 | 0.0340 | **void**: both below 10% |
| PickCube | +14.7 pp | 1.80e-8 | 7.21e-8 | significant win |
| PokeCube | +3.5 pp | 0.103 | 0.207 | inconclusive |
| PullCube | +45.5 pp | 6.75e-49 | 4.73e-48 | significant win |
| PushCube | -0.8 pp | 0.795 | 0.795 | tie/inconclusive |
| RollBall | +40.0 pp | 2.67e-36 | 1.60e-35 | significant win |
| StackCube | +14.0 pp | 1.95e-9 | 9.74e-9 | significant win |

Result: **five significant non-void wins, zero significant losses**. This clears the exact preregistered H1 rule.

### H2: QP-learned versus non-learned QP-anchor — **strong positive evidence for learning**

The same five tasks—AnymalC, PickCube, PullCube, RollBall and StackCube—remain Holm-significant wins. PokeCube is inconclusive, PushCube is a tie/slight loss, and LiftPeg is void. This establishes that a learned shape reference matters relative to the fixed anchor. It does **not** establish that the QP projection matters.

### H3: QP-learned versus standalone MLP — **mixed and damaging to a general method claim**

| Task | QP-learned minus MLP | Holm p | Interpretation |
|---|---:|---:|---|
| AnymalC-Reach | -0.5 pp | 1.000 | tie |
| LiftPegUpright | -82.7 pp | 3.05e-97 | MLP decisively better |
| PickCube | -25.6 pp | 1.13e-12 | MLP decisively better |
| PokeCube | -7.0 pp | 0.00247 | MLP significantly better |
| PullCube | -0.7 pp | 1.000 | void/tie |
| PushCube | -31.8 pp | 4.74e-30 | MLP decisively better |
| RollBall | +37.3 pp | 6.89e-28 | governor decisively better |
| StackCube | +1.5 pp | 1.000 | tie |

This is the main causal problem. The governor clearly adds value over the standalone MLP only on RollBall. On four tasks the MLP is significantly better, and on the remaining tasks the comparison is tied or void.

### H4: QP-learned versus learned raw output — **does not validate the QP mechanism**

The preregistered saturation-heavy family contains AnymalC, LiftPeg, PullCube and RollBall. No non-void comparison survives Holm correction. PullCube has an adjusted p=0.0325, but both methods exceed 90%, so the preregistration marks it void. The box/conservation projection therefore lacks confirmatory task-success evidence over the raw learned decoder.

### H6: QP-learned versus learned tanh — **FAIL as a superiority mechanism**

No task has a Holm-significant governor win. Learned-tanh is numerically better on AnymalC, LiftPeg, PickCube, PokeCube, PullCube and RollBall; QP-learned is numerically better only on PushCube and StackCube, without corrected significance.

This is stronger evidence than “the effect is uncertain”: the proposed QP mechanism has not demonstrated a success-rate advantage over a simpler bounded learned decoder.

## 5. Native-rate and simple-baseline context

- QP-learned remains significantly below native execution on seven tasks after Holm correction; PullCube is a void near-ceiling tie.
- This does not invalidate a rate-conversion paper: native execution is the reference ceiling, not a competing 4X reconstruction method.
- It forbids language such as “no-loss rate conversion,” “native-equivalent,” or “general recovery.”
- QP-learned significantly beats ZOH on AnymalC, PickCube, PullCube, RollBall and StackCube. PokeCube misses Holm significance, PushCube is a nonsignificant loss, and LiftPeg is void.

## 6. Independent evidence from the non-learned governor

The explicit n=400 locked comparison uses only paired arms present in each source file:

| Task / rate | Native | ZOH | TAC-Fold | QP | QP-anchor | QP-anchor vs TAC-Fold | QP-anchor vs QP |
|---|---:|---:|---:|---:|---:|---:|---:|
| PickCube, 4X | 88.0% | 40.5% | 50.2% | 52.2% | 53.5% | +3.2 pp, p=0.0106 | +1.2 pp, p=0.424 |
| PushT, 2X | 24.5% | 11.2% | 21.5% | 18.2% | 21.5% | +0.0 pp, p=1.000 | +3.2 pp, p=0.171 |

Interpretation: constraint-preserving anchoring produces a real but small PickCube improvement and no PushT improvement. This independently supports a **task-conditional** mechanism, not a general one.

Sources:

- `result_dp_PickCube-v1_k4_n400_anchor.json`
- `result_dp_PushT-v1_fair_n400.json`
- `PAPER_LOCKED_RESULTS.md`
- `paper_locked_results_manifest.json`

## 7. Provenance and reproducibility evidence

All confirmation files record held-out evaluation protocol, training budget, checkpoint hash and raw per-episode booleans.

| Task | n | Evaluation offset | Result SHA-256 | Checkpoint SHA-256 prefix |
|---|---:|---:|---|---|
| AnymalC-Reach | 400 | 100 | `c34c4dbc4e42...` | `2b228db93cd6...` |
| LiftPegUpright | 400 | 100 | `ae8a05972198...` | `aceac7373fbc...` |
| PickCube | 293 | 500 | `b739b3be9464...` | `42e9aa32a0c1...` |
| PokeCube | 400 | 100 | `a711f8eca56d...` | `3723924a3fd3...` |
| PullCube | 400 | 100 | `5f6eef5413e2...` | `7f6583617560...` |
| PushCube | 400 | 100 | `f20e5173f223...` | `9fa682dbd336...` |
| RollBall | 400 | 100 | `b7c74b202403...` | `7e47040b171c...` |
| StackCube | 400 | 100 | `1211032c2705...` | `6dcdd2eb8bcb...` |

All use 200 training demonstrations, 30,000 policy-training steps and held-out demonstration states. The eight result files contain 3,093 paired episodes per arm and 24,744 arm-episode outcomes total.

### Known provenance defect

The legacy `assemble_paper_table.py` selects the largest arm result found through filename globbing. It can mix development windows, legacy protocols and different sample sizes. Its output is not publication-safe. Only explicit-source tables with a manifest should be used.

## 8. Prior-art boundary

Three current papers prevent a broad novelty claim:

- **CAT** already presents a trajectory-level continuous representation with frequency-aware coordinates and evaluations across control frequencies: <https://arxiv.org/abs/2608.24111>
- **Spline Policy** replaces action chunks with learned spline parameters that can be queried at different temporal resolutions and integrated with controllers: <https://arxiv.org/abs/2606.07386>
- **B-Spline Policy** predicts B-spline trajectories that can be temporally scaled and executed at higher rates: <https://arxiv.org/abs/2607.09648>

Consequences:

- Do not claim the first multi-rate policy, continuous action representation, temporally scalable representation or spline-like decoder.
- The remaining gap is narrower: exact block-displacement preservation under bounded reconstruction, coupled to a learned shape reference.
- Because that constraint mechanism did not beat learned-tanh, the paper cannot currently sell the gap as an empirically superior method.

## 9. Evidence missing for a decent Q2 submission

These are publication gates, not optional polish.

### Gate A — matched external baseline

Train and evaluate at least B-Spline Policy under the same backbone, demonstrations, 30k budget, action horizon and confirmation episodes. The repository already has a `--head bspline` route, but it must be validated against the official algorithm rather than merely named similarly.

Minimum acceptable report:

- all eight tasks at `k=4`;
- identical paired episode windows;
- at least the same training seeds used for the proposed method;
- exact McNemar comparisons with a sealed Holm family;
- no augmentation of the external method with the authors' own `satfix` unless reported as a separate hybrid.

### Gate B — training-seed replication

The current p-values measure paired episode disagreement for one trained checkpoint per task. They do not show that another training seed would preserve the ranking. A decent Q2 submission should use at least three independently trained seeds for the proposed learned method and learned external baseline on the primary tasks. Episode pooling must not be presented as independent model replication.

### Gate C — claim rewrite

Recommended claim:

> Learned action-shape reconstruction substantially reduces control-rate conversion failure on selected task regimes; exact conservation provides deterministic interface guarantees but does not consistently improve task success over simpler bounded learned decoding.

This is weaker than the original method claim but matches the evidence.

### Gate D — artifact repair

- Regenerate every manuscript table from explicit source manifests.
- Remove or quarantine legacy mixed-protocol tables.
- Tie every prose number to a source JSON and hash.
- Report all eight tasks, including PushCube and LiftPeg failures.
- Call the evaluated model a state-based robot policy, not a VLA.

### Gate E — generality disclosure

The current evidence is simulation-only, state-only and one policy family. For RA-L, at least one visual-policy or physical-robot confirmation would materially improve survival. If unavailable, state this prominently and target the manuscript as a scoped simulation study rather than a general VLA method.

## 10. Q2 decision rule

### Credible Q2/RA-L method paper

Proceed as a positive method paper only if:

1. the matched learned spline baseline is implemented faithfully;
2. the learned reconstructor remains competitive across independent training seeds;
3. the manuscript abandons QP-superiority language unless a new untouched experiment establishes it; and
4. all tables pass explicit-source provenance checks.

### Credible Q2/TMLR diagnostic paper

If the external baseline matches or beats the governor, the evidence can still support a diagnostic paper:

> Control-rate conversion failures are large and task-dependent; learned reconstruction helps, while conservation constraints supply guarantees but not universal success gains.

This is more honest and probably more defensible than forcing a failed universal mechanism claim.

### Do not submit yet if

- the paper still claims general governor superiority;
- only the favorable five H1 tasks are shown;
- episode-level p-values are presented as seed robustness;
- `learned_tanh` and standalone MLP are omitted;
- the external baseline is only post-hoc interpolation or an internally modified B-spline arm;
- legacy mixed-protocol tables remain in the manuscript.

## 11. Final verdict

The project has enough real evidence to justify continued publication work. It does **not** yet have enough evidence for a decent Q2 method-paper submission. The decisive positive fact is the preregistered 5/8 win against TAC-Fold. The decisive negative fact is that the proposed QP mechanism does not beat learned-tanh and is frequently beaten by a standalone MLP. The decisive missing fact is performance against a faithfully matched published learned spline representation across training seeds.

Complete the external baseline and seed gate, then submit with the narrow claim. Without those two additions, a competent reviewer can reject the paper on causality and comparator adequacy without disputing any of the reported statistics.
