# Paper Outline

## Structure Pattern: IMRaD Method Paper

## Overview

The paper presents ARC--TAC as a compact action-head interface for one imitation policy queried at several declared action resolutions. The introduction defines action resolution separately from controller frequency, inference latency, and wall-clock speed. Related work then positions CAT, Spline Policy, and B-Spline Policy as the closest prior art and limits novelty to the combination of explicit resolution conditioning, block-integral supervision, and conservative bounded reconstruction. The method section formalizes that combination and its conservation, identity, and feasibility properties. Results follow a preregistered progression from Push-T through RoboMimic and RoboCasa, ending with an RGB-plus-proprioception confirmation; all central empirical claims remain contingent on the locked campaign outputs. The discussion interprets only supported positive effects, reports failure cells and limitations, and does not claim language conditioning or faster task completion.

## Detailed Outline

### Abstract (~250 words, excluded from the 8,000-word body)

**Purpose:** State the action-resolution problem, the coupled ARC--TAC method, the evaluation scope, the principal preregistered result, and the exact claim boundary.

**Content summary:** Define ARC as Action-Resolution Conditioning and TAC as the conservative controller-rate reconstruction layer. Report only final verified effect sizes and confidence intervals from the locked analysis, followed by the formal guarantees and the visual-policy confirmation. End by stating that the intervention concerns action representation and reconstruction, not language conditioning or wall-clock acceleration.

**Sources:** Diffusion Policy; CAT; Spline Policy; B-Spline Policy; RoboMimic; RoboCasa.

**Key arguments:** A single resolution-conditioned policy with conservative reconstruction reduces multi-resolution degradation against matched post-hoc spline and B-spline baselines while preserving predicted block displacement.

**Transition to Section 1:** The abstract's compact claim motivates a precise definition of the deployment mismatch and why existing temporal representations leave a conservative-interface gap.

### 1. Introduction (~900 words)

**Purpose:** Establish the practical problem, define the paper's terms, identify the narrow gap, and state contributions and research questions without overclaiming.

**Content:**

- **1.1 Action interfaces are a deployment variable**
  - Explain why temporal action abstraction affects learnability, stability, and reuse across robot-control stacks.
  - Motivate one policy serving declared action resolutions without per-resolution retraining.
- **1.2 Scope and semantic boundary**
  - Define native controller steps, action blocks, action resolution, and reconstruction.
  - Distinguish the intervention from spectral frequency, asynchronous inference, replanning horizon, and wall-clock acceleration.
- **1.3 Gap and research questions**
  - Recognize CAT and spline-based policies as direct predecessors.
  - Ask whether joint resolution conditioning and conservative reconstruction improves success over native heads, cubic spline, B-spline, ZOH, and post-hoc TAC controls.
- **1.4 Contributions**
  - A fixed-count resolution-conditioned block-integral target.
  - A cumulative-path TAC decoder with exact block-displacement and bound-feasibility properties.
  - A preregistered matched-backbone evaluation on Push-T, RoboMimic, RoboCasa, plus one visual observation cell.

**Sources:** Feng et al. (action-space design); Diffusion Policy; ACT; Open X-Embodiment; FAST; CAT; Spline Policy; B-Spline Policy; RTC; REMAC.

**Key arguments:** Action resolution is consequential but semantically distinct from latency and speed. The contribution is the coupled interface and evidence, not the first temporal or resampleable action representation.

**Transition to Section 2:** Having bounded the claim, the related-work taxonomy shows precisely which capabilities are inherited, adjacent, or absent in prior methods.

### 2. Related Work (~1,050 words)

**Purpose:** Establish prior art, identify the closest competitors, and derive the comparison requirements used in the experiment design.

**Content:**

- **2.1 Action chunking and generative action heads**
  - Cover ACT, Diffusion Policy, consistency policies, and flow-based action experts.
  - Separate chunk prediction benefits from reconstruction effects.
- **2.2 Frequency-aware and continuous trajectory representations**
  - Treat CAT as the closest conceptual predecessor.
  - Compare DMP temporal scaling, FAST tokenization, FreqPolicy, Spline Policy, and B-Spline Policy.
  - State that continuous/resampleable representations and frequency awareness are established ideas.
- **2.3 Execution-time adaptation and latency**
  - Taxonomize RTC, TAS, PACE, and REMAC as asynchronous, selection, or prefix-execution methods.
  - Explain why their speed/latency outcomes are not outcomes of the present action-resolution intervention.
- **2.4 Residual gap and required controls**
  - Identify exact block-displacement preservation after bounded reconstruction as the residual technical gap.
  - Motivate matched cubic-spline, B-spline, ZOH, and post-hoc TAC controls and component ablations.

**Sources:** Akima; Ijspeert et al.; Diffusion Policy; ACT; Consistency Policy; $\pi_0$; FAST; RTC; FreqPolicy; TAS; CAT; Spline Policy; B-Spline Policy; PACE; REMAC; Lazzati et al.

**Key arguments:** Prior methods establish chunking, continuous trajectories, temporal scaling, and latency-aware execution; none in the screened set combines ARC's targets with TAC's exact conservative decoder.

**Transition to Section 3:** The residual gap determines the method: the learned target must have a conserved physical meaning, and the decoder must preserve it under temporal refinement and feasible saturation.

### 3. Method (~1,850 words)

**Purpose:** Specify ARC--TAC sufficiently for reproduction and prove the limited invariants that support its mechanism claim.

**Content:**

- **3.1 Problem formulation and notation**
  - Define native action sequence, continuous delta-action dimensions, held dimensions, block size, declared resolution set, policy observation, and receding-horizon execution.
  - Define the fixed prediction count and changing native-time coverage.
- **3.2 Action-Resolution Conditioning (ARC)**
  - Construct block-mean targets for continuous dimensions and causal first-value targets for held dimensions.
  - Convert block means to block sums before reconstruction.
  - Encode the rate condition and train one diffusion policy jointly over the declared resolutions.
- **3.3 Temporal Action Conservation (TAC) folding**
  - Interpolate cumulative block endpoints with the local Akima-style path used by TAC-Fold.
  - Difference the refined cumulative path into controller-step actions.
  - Describe unconstrained TAC, saturation correction, and handling of held dimensions.
- **3.4 Formal properties**
  - Proposition 1: blockwise displacement conservation by telescoping differences.
  - Proposition 2: native-resolution identity at block size one.
  - Proposition 3: existence and correctness of the bounded projection when the block sum lies within aggregate bounds.
  - Clarify that these are representation/decoder guarantees, not closed-loop stability guarantees.
- **3.5 Baselines and ablations**
  - Native step head; ZOH; post-hoc TAC; raw cubic spline; bounded-error B-spline; ARC with each decoder; unconstrained versus saturation-corrected TAC.
  - Keep the backbone, demonstrations, training steps, execution horizon, seeds, and controller semantics matched.
- **3.6 Reproducibility and implementation**
  - Report architecture, optimization, rate sampling, target normalization, checkpoints and hashes, software versions, random seeds, and artifact paths.
  - Describe state-policy and RGB-plus-proprioception branches without calling the latter language-conditioned.

**Sources:** Akima; Ijspeert et al.; Diffusion Policy; ACT; CAT; Spline Policy; B-Spline Policy; Feng et al.; Lazzati et al.

**Key arguments:** Block-integral supervision supplies a resolution-stable target meaning, while TAC makes conservation explicit rather than relying on smooth interpolation to preserve displacement.

**Transition to Section 4:** The formal construction yields three empirical questions: whether the coupled method improves task success, whether both components matter, and whether the effect extends beyond state inputs.

### 4. Experimental Design (~1,100 words)

**Purpose:** Predeclare datasets, comparisons, estimands, statistical tests, and acceptance criteria so positive claims are traceable to a fixed protocol.

**Content:**

- **4.1 Benchmarks and data**
  - Push-T: 200 demonstrations, 400 paired evaluation episodes, 30k training steps.
  - RoboMimic Lift, Can, Square: 200 demonstrations, 100 paired evaluation episodes per task, 30k steps.
  - RoboCasa: four manipulation tasks, 39 demonstrations, 100 random-reset evaluations per task, 15k steps.
  - Visual confirmation: RoboCasa CloseSingleDoor, RGB plus proprioception, 35 demonstrations, 15 paired evaluations, 15k steps.
- **4.2 Comparisons and factorial decomposition**
  - Evaluate declared resolutions $k\in\{1,2,4\}$.
  - Separate step-head post-hoc reconstruction from joint ARC training and TAC decoding.
  - Name cubic spline and B-spline as required primary comparator families.
- **4.3 Outcomes and statistical analysis**
  - Primary outcome: paired closed-loop task success.
  - Exact McNemar tests for binary paired outcomes; Holm correction over six preregistered primary comparisons.
  - Paired bootstrap interval for native-resolution no-harm; report absolute paired differences and episode counts.
- **4.4 Claim gates and integrity controls**
  - Primary positive claim requires ARC+TAC at $k=2$ to beat post-hoc TAC, cubic spline, and B-spline on Push-T and pooled non-void manipulation.
  - Mechanism requires decoder/representation ablations and conservation diagnostics.
  - Validate budgets, result provenance, checkpoint hashes, and exclusions before analysis.

**Sources:** Diffusion Policy; RoboMimic; MimicGen; RoboCasa; CAT; Spline Policy; B-Spline Policy; Consistency Policy.

**Key arguments:** Paired evaluation and matched training isolate the action interface; the preregistered six-comparison family prevents selective positive reporting.

**Transition to Section 5:** With estimands and gates fixed, the results can be presented from primary tests to mechanism and generality without changing the success criterion after observation.

### 5. Results (~1,750 words)

**Purpose:** Report the complete preregistered evidence, foregrounding the positive result while preserving all denominators, uncertainty, corrected tests, and contrary cells.

**Content:**

- **5.1 Primary positive comparisons**
  - Table of ARC+TAC versus post-hoc TAC, cubic spline, and B-spline at $k=2$ for Push-T and pooled manipulation.
  - Report paired wins/losses, absolute success-rate differences, exact $p$ values, Holm-adjusted decisions, and checkpoint provenance.
  - Use final numerical text only after the analyzer passes all integrity checks.
- **5.2 Benchmark-level robustness**
  - Break down Push-T, RoboMimic Lift/Can/Square, and four RoboCasa tasks over $k=1,2,4$.
  - Show full cells, including null or negative effects, while distinguishing primary from supporting analyses.
- **5.3 Component and mechanism analysis**
  - Compare ARC+ZOH, ARC+spline, ARC+B-spline, ARC+TAC, and step-head post-hoc variants.
  - Report block-displacement residuals, saturation events, smoothness diagnostics, and any relationship to success.
- **5.4 Native-resolution no-harm and visual confirmation**
  - Report the preregistered lower confidence bound for $k=1$ relative to the native head.
  - Report CloseSingleDoor RGB-plus-proprioception outcomes as a scoped backbone/observation confirmation, not a VLA or language result.
- **5.5 Sensitivity and failure cases**
  - Identify task/resolution cells where ARC--TAC does not improve or saturation assumptions are stressed.
  - Distinguish exploratory sensitivity checks from confirmatory results.

**Sources:** Original campaign artifacts; Diffusion Policy for Push-T context; RoboMimic; RoboCasa; CAT; Spline Policy; B-Spline Policy for comparator interpretation.

**Key arguments:** The section may assert a positive method result only if the fixed primary gates pass; otherwise the claim must be narrowed and any method iteration documented as an amendment and independently reevaluated.

**Transition to Section 6:** The complete result pattern supports interpretation of where conservation helps, how it differs from spline smoothness, and what remains unestablished.

### 6. Discussion (~1,000 words)

**Purpose:** Interpret the evidence relative to the mechanism and prior art, explain practical impact, and state limitations that constrain generalization.

**Content:**

- **6.1 Interpretation of the coupled effect**
  - Attribute gains only as far as the factorial comparisons allow.
  - Contrast conserved displacement with smoothness-only explanations and the known benefits of action chunking.
- **6.2 Relationship to CAT and spline policies**
  - Present ARC--TAC as a simpler conservative interface, not a replacement for richer continuous latent or spline representations.
  - Explain what the matched spline/B-spline baselines establish and what the absence of a full CAT reproduction leaves unresolved.
- **6.3 Practical impact**
  - Discuss reuse of one action head across declared resolutions, controller integration, and diagnostic invariants.
  - Avoid extrapolation to faster completion, cross-embodiment transfer, or language-conditioned policies.
- **6.4 Limitations, threats, and future work**
  - Simulation-only evidence; limited visual sample; no real robot; no language input; fixed resolution set; delta-action assumptions; no closed-loop stability theorem.
  - Prioritize head-to-head CAT/Spline Policy reproduction, real-time hardware, additional backbones, language-conditioned VLA testing, and adaptive resolution selection.

**Sources:** CAT; Spline Policy; B-Spline Policy; FAST; RTC; REMAC; PACE; TAS; FreqPolicy; Lazzati et al.; Open X-Embodiment; RT-1; RT-2; Octo; OpenVLA; $\pi_0$.

**Key arguments:** The value is a measurable conservative interface with explicit invariants and modest integration cost; generality beyond the tested action heads and simulators remains future work.

**Transition to Section 7:** The limitations narrow the final takeaway to the empirically supported and formally guaranteed contribution.

### 7. Conclusion (~350 words)

**Purpose:** Restate the verified contribution, evidence, and scope in a concise closing argument.

**Content:**

- Reanswer the central research question using the final preregistered statistics.
- Summarize the coupled novelty: resolution condition, block-integral target, and conservative bounded reconstruction.
- Close with the practical implication and the next validation step, without introducing new evidence.

**Sources:** Synthesis of original results; CAT; Spline Policy; B-Spline Policy.

**Key arguments:** ARC--TAC is supported as a partially novel, practically useful multi-resolution action interface within the evaluated domains and declared semantic boundaries.

**Transition:** End of paper; references and appendices provide source provenance, proofs, extended tables, and reproducibility details.

## Evidence Map

| Source | Assigned section(s) | Role and stance |
|---|---|---|
| Akima (1970) | 2.2, 3.3 | Foundational interpolation provenance; neutral |
| Ijspeert et al. (2013) | 2.2, 3.1 | Temporal-scaling precedent; opposes broad novelty |
| Mandlekar et al. (2021), RoboMimic | Abstract, 4.1, 5.2 | Benchmark and protocol support; neutral |
| Brohan et al. (2022), RT-1 | 6.4 | Generalist/VLA context; neutral |
| Chi et al. (2023), Diffusion Policy | Abstract, 1.1, 2.1, 3.2, 4.1, 5.2 | Backbone and Push-T provenance; neutral/supporting |
| Zhao et al. (2023), ACT | 1.1, 2.1, 3.2 | Action-chunking precedent; opposes broad novelty |
| Brohan et al. (2023), RT-2 | 1.2, 6.4 | Defines VLA claim boundary; opposes language overclaim |
| Mandlekar et al. (2023), MimicGen | 4.1 | Demonstration/data-generation context; neutral |
| Open X-Embodiment Collaboration et al. (2023) | 1.1, 6.3 | Heterogeneous interface motivation; supporting |
| Nasiriany et al. (2024), RoboCasa | Abstract, 4.1, 5.2, 5.4 | Benchmark and visual-cell provenance; neutral |
| Prasad et al. (2024), Consistency Policy | 2.1, 4.4 | Inference-speed boundary; opposes acceleration conflation |
| Octo Model Team et al. (2024) | 6.4 | Flexible generalist-policy context; neutral |
| Kim et al. (2024), OpenVLA | 6.4 | Future VLA-backbone context; neutral |
| Black et al. (2024), $\pi_0$ | 2.1, 6.4 | Flow action-head/VLA context; neutral |
| Pertsch et al. (2025), FAST | 1.1, 2.2, 6.2 | High-frequency tokenization prior art; opposes broad novelty |
| Black et al. (2025), RTC | 1.2, 2.3, 6.3 | Asynchronous latency boundary; opposes speed conflation |
| Su et al. (2025), FreqPolicy | 2.2, 6.4 | Spectral/computational frequency distinction; neutral |
| Weng et al. (2025), TAS | 2.3, 6.4 | Chunk-selection execution context; neutral |
| Yang et al. (2026), CAT | Abstract, 1.3, 2.2, 2.4, 3.2, 4.2, 5.1, 6.2, 7 | Closest predecessor; opposes broad novelty, supports narrow gap |
| Tian et al. (2026), Spline Policy | Abstract, 1.3, 2.2, 2.4, 3.5, 4.2, 5.1, 6.2, 7 | Closest continuous-trajectory competitor; opposes broad novelty |
| Han et al. (2026), B-Spline Policy | Abstract, 1.3, 2.2, 2.4, 3.5, 4.2, 5.1, 6.2, 7 | Required temporal-scaling competitor; opposes broad novelty |
| Nie et al. (2026), PACE | 2.3, 6.4 | Adaptive prefix-execution context; neutral |
| Wang et al. (2026), REMAC | 1.2, 2.3, 6.3 | Delay-robust execution boundary; opposes latency conflation |
| Feng et al. (2026) | 1.1, 3.5 | Significance of action-space design; supporting |
| Lazzati et al. (2026) | 2.1, 3.5, 6.1 | Mechanism caution and ablation rationale; opposing simplistic attribution |
| Original ARC--TAC artifacts | 3, 4, 5, 6 | Method, proofs, preregistration, code, checkpoints, and empirical evidence; claim-determining |

## Transition Logic Summary

| Boundary | Required reader state and connection |
|---|---|
| Abstract $\rightarrow$ Introduction | Recognize the claimed positive result; learn the exact problem and boundary next. |
| Introduction $\rightarrow$ Related Work | Understand the narrow claim; test it against the strongest prior art. |
| Related Work $\rightarrow$ Method | See that conservation and target/decoder coupling are the residual gap; formalize them. |
| Method $\rightarrow$ Experimental Design | Understand what each component changes; isolate those changes empirically. |
| Experimental Design $\rightarrow$ Results | Know the locked comparisons, outcomes, and gates before seeing numbers. |
| Results $\rightarrow$ Discussion | Carry the full pattern, including negative cells, into mechanism and impact interpretation. |
| Discussion $\rightarrow$ Conclusion | Reduce the implications to the exact supported novelty and scope. |

## Word Count Summary

| Section | Target words | Share of 8,000-word body |
|---|---:|---:|
| Introduction | 900 | 11.25% |
| Related Work | 1,050 | 13.125% |
| Method | 1,850 | 23.125% |
| Experimental Design | 1,100 | 13.75% |
| Results | 1,750 | 21.875% |
| Discussion | 1,000 | 12.5% |
| Conclusion | 350 | 4.375% |
| **Total body** | **8,000** | **100%** |

Abstract target: 250 words, excluded from the body count. References and appendices are excluded.

## Phase 2 Quality-Gate Status

- Recognized IMRaD structure with original experiments: **pass**.
- Every major section has a purpose, content summary, sources, arguments, and transition: **pass**.
- Corrected body allocation totals 8,000 words: **pass**.
- All 25 Phase 1 sources are assigned: **pass**.
- Every adjacent section boundary has transition logic: **pass**.
- Heading depth is at most three levels: **pass**.
- User approval: **pass** — user directed “FIX, NO STOPPING” on 2026-09-18 after the outline-approval gate was presented.
