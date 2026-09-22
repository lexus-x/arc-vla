# Argument Blueprint

## Central Thesis

This paper argues that ARC--TAC is a useful conservative action interface for a single imitation policy queried at several declared action resolutions because (i) ARC gives each predicted token an explicit resolution-dependent block-integral meaning, (ii) TAC reconstructs bounded controller-rate commands while preserving every feasible predicted block displacement, and (iii) the preregistered paired campaign is designed to test whether this coupling reduces closed-loop degradation relative to native step heads and matched cubic-spline, B-spline, ZOH, and post-hoc TAC controls. The empirical portion of this thesis becomes assertive only if the locked primary and mechanism gates pass.

## Claim Status Vocabulary

| Status | Meaning in this paper |
|---|---|
| Proven | Follows from stated assumptions and is covered by executable tests. |
| Established prior art | Supported by a cited primary source; not claimed as novel. |
| Confirmatory pending | Has a fixed estimator and acceptance gate but no completed campaign evidence yet. |
| Supporting pending | Useful for scope or mechanism, but not part of the primary family. |
| Out of scope | Must not appear as an achieved contribution. |

## Sub-Arguments

### Sub-Argument 1: Action resolution is a real policy-interface variable, but broad frequency-aware representation claims are occupied prior art

- **Claim:** Changing the temporal meaning of action outputs can alter learnability and closed-loop behavior, so deployment across action resolutions is a substantive representation problem rather than a formatting detail.
- **Evidence:** Feng et al. systematically vary temporal and spatial action abstractions and find material policy effects; Diffusion Policy and ACT show that multi-step action prediction is itself consequential; CAT explicitly targets shared representations across control frequencies.
- **Evidence:** DMPs, FAST, Spline Policy, and B-Spline Policy establish temporal scaling, frequency-space tokenization, and continuously queryable trajectory representations.
- **Reasoning:** These independent lines justify the problem's significance while ruling out “first multi-rate,” “first frequency-aware,” and “first resampleable action policy” claims. The defensible gap must therefore be narrower than temporal scaling itself.
- **Counter-argument:** ARC is merely scalar conditioning and block averaging, a trivial special case of CAT or spline trajectory tokens.
- **Rebuttal:** **Concede and limit.** Neither scalar conditioning nor block averaging alone is claimed as a major novelty. The paper claims partial novelty only for their coupling to block-integral semantics and a bounded exact-displacement decoder, then requires component ablations to show practical value beyond post-hoc interpolation.
- **Status:** Established motivation and novelty boundary; source-backed.

### Sub-Argument 2: ARC assigns a conserved, resolution-dependent physical meaning to fixed-count action outputs

- **Claim:** For declared resolution (k\in\{1,2,4\}), an ARC continuous-action token is a block mean over (k) native demonstration actions, so multiplying it by (k) recovers the target block displacement exactly.
- **Evidence:** Method definition in `heads.arc_targets`; executable tests cover all declared rates; training supplies the normalized condition (r=\log_2(k)/2) to one shared policy.
- **Reasoning:** A fixed output count can cover different native-time spans without changing the meaning covertly: the condition declares the resolution, while the target construction makes each token's integral explicit. This differs from applying a test-time interpolator to a step head that was trained only at native resolution.
- **Counter-argument:** The network could ignore the scalar rate condition, making ARC equivalent to a mixed-target training artifact.
- **Rebuttal:** **Test rather than assume.** Report rate-conditioned target losses and closed-loop factorial comparisons. ARC+ZOH versus step-head+ZOH isolates joint-rate learning; ARC+TAC versus ARC+ZOH isolates decoding. No representation claim survives if the matched ablations do not support it.
- **Status:** Target identity proven; learned use of the condition confirmatory pending.

### Sub-Argument 3: TAC provides exact feasible displacement conservation, bounded commands, and native-resolution identity

- **Claim:** For every continuous block with bounded ARC mean, TAC-Fold with saturation repair returns controller-step commands in ([-1,1]) whose blockwise sum equals the commanded block displacement; at (k=1), decoding is the identity on the bounded predicted mean.
- **Evidence:** Cumulative-path construction in `resample_tac_fold`, block repair in `_project_block`, ARC mean bounding in `heads.decode`, and regression tests at (k=1,2,4), including out-of-range raw network outputs.
- **Reasoning:** TAC samples a Hermite path whose endpoints are cumulative block sums. Differencing sampled positions telescopes to the endpoint difference. Bounding the predicted mean before forming (S=k\bar\mu) guarantees (|S_d|\le k), the necessary and sufficient coordinatewise feasibility condition. Iterative clip-and-redistribute repair preserves the target sum while enforcing each command bound.
- **Counter-argument:** Any cumulative spline also telescopes, so TAC's conservation is not distinctive.
- **Rebuttal:** **Concede and differentiate.** Endpoint interpolation provides unconstrained conservation for several curve families. The claimed distinction is not telescoping alone; it is the explicit target/decoder contract plus a feasibility-guaranteed bounded repair. Raw cubic-spline and B-spline controls test whether smooth interpolation without that repair is sufficient.
- **Counter-argument:** A blockwise invariant does not imply closed-loop stability or task success.
- **Rebuttal:** **Acknowledge as limitation.** The propositions are interface guarantees only. Closed-loop utility is an empirical claim evaluated by task success, and no Lyapunov or stability theorem is asserted.
- **Status:** Proven under explicit assumptions; closed-loop consequence confirmatory pending.

### Sub-Argument 4: The preregistered campaign can causally separate joint-rate learning from reconstruction choice within the evaluated backbone

- **Claim:** Matched training and paired rollouts across Push-T, RoboMimic, and RoboCasa can attribute differences to the action representation/decoder more cleanly than comparisons that change backbone, data, and execution protocol together.
- **Evidence:** The preregistration fixes demonstrations, steps, rates, episode counts, observation modes, controller semantics, checkpoints, paired evaluation seeds/noise, six primary comparisons, exact McNemar tests, and Holm correction.
- **Evidence:** Step-head+post-hoc TAC, step-head+cubic spline, step-head+B-spline, ARC+ZOH, and ARC+TAC form the required factorial contrasts.
- **Reasoning:** Within each paired cell, the initial condition and evaluation protocol are matched. Across the declared contrasts, the training representation and decoder vary separately enough to test whether either component alone explains the effect.
- **Counter-argument:** The spline implementations are reconstruction baselines, not full reproductions of Spline Policy, B-Spline Policy, or CAT.
- **Rebuttal:** **Concede and label precisely.** Call them matched cubic-spline and Algorithm-1-style bounded-error B-spline reconstruction controls, never full policy reproductions. Treat the absence of a full CAT/head-to-head reproduction as a limitation.
- **Counter-argument:** Pooling heterogeneous manipulation tasks can hide failures.
- **Rebuttal:** **Report both levels.** The pooled manipulation comparison is preregistered for power, while every task/rate cell, void status, denominator, and discordant pair is shown separately.
- **Status:** Design strength established; result pending.

### Sub-Argument 5: A positive result would have scoped practical impact without constituting a VLA, speed, or universal-superiority result

- **Claim:** If the gates pass, ARC--TAC supports reuse of one action head across the tested action resolutions and simulators, with a visual-observation confirmation on RoboCasa CloseSingleDoor.
- **Evidence:** Primary cells cover Push-T, RoboMimic Lift/Can/Square, and four RoboCasa tasks; the separate visual cell uses RGB plus proprioception with the same ARC head and decoder.
- **Reasoning:** Breadth across task families and one visual input path supports an interface-level engineering result. It does not establish language conditioning, cross-embodiment transfer, real-robot safety, or faster task completion.
- **Counter-argument:** State-policy evidence plus one small visual cell is too weak for a “VLA method” claim.
- **Rebuttal:** **Agree.** Describe ARC--TAC as compatible with chunked visual-action/VLA action heads, but call the experiment a visual-policy validation and explicitly state that no language input was evaluated.
- **Counter-argument:** Simulator action-resolution changes may be presented as acceleration.
- **Rebuttal:** **Reject the inference.** Completion time and asynchronous latency are not estimands. RTC, REMAC, PACE, and Consistency Policy are cited to keep this boundary explicit.
- **Status:** Impact claim confirmatory pending and deliberately scoped.

## Formal Claim Ledger

### Definition 1: ARC target

For native continuous actions (a_{t,d}\in[-1,1]), block (b), and declared resolution (k),

\[
\mu_{b,d}^{(k)}=\frac{1}{k}\sum_{j=0}^{k-1}a_{bk+j,d},
\qquad
S_{b,d}^{(k)}=k\mu_{b,d}^{(k)}.
\]

Trailing held dimensions use the block's first command and are outside the continuous-displacement propositions.

### Lemma 1: ARC target conservation

**Statement:** (S_{b,d}^{(k)}=\sum_{j=0}^{k-1}a_{bk+j,d}).

**Proof obligation:** Direct substitution into Definition 1.

**Executable evidence:** `test_arc_targets_and_tacfold_preserve_predicted_block_displacement` at all declared rates.

### Proposition 1: TAC endpoint conservation

Let cumulative endpoints satisfy (C_0=0) and (C_{b+1}=C_b+S_b). Let the reconstructed commands be differences (u_{bk+j}=\widetilde C_{bk+j+1}-\widetilde C_{bk+j}) of any interpolated cumulative path that includes both block endpoints. Then

\[
\sum_{j=0}^{k-1}u_{bk+j}=S_b.
\]

**Proof obligation:** The finite sum telescopes to (\widetilde C_{(b+1)k}-\widetilde C_{bk}=C_{b+1}-C_b=S_b).

**Scope:** Exact in real arithmetic; reported implementation residual is subject to floating-point tolerance.

### Proposition 2: Feasible bounded repair

For each coordinate, a sequence (v_1,\ldots,v_k\in[-1,1]) with sum (S) exists if and only if (|S|\le k). ARC forms (S=k\,\mathrm{clip}(\widehat\mu,-1,1)), so feasibility always holds. The clip-and-redistribute repair terminates with every (v_j\in[-1,1]) and (\sum_jv_j=S), up to the declared numerical tolerance.

**Proof obligation:** Necessity follows from the triangle inequality; sufficiency follows from the constant feasible vector (v_j=S/k). For the implemented repair, a redistribution either closes the deficit or causes at least one additional coordinate to reach a bound; with at most (k) coordinates, the process terminates in finitely many steps under exact arithmetic.

**Executable evidence:** `test_arc_satfix_bounds_and_conserves_out_of_range_predictions` at (k=2,4), plus the all-rate conservation test.

### Proposition 3: Native-resolution identity

At (k=1), the cumulative path has one reconstructed difference per block, hence (u_b=S_b=\mathrm{clip}(\widehat\mu_b,-1,1)); bounded repair leaves it unchanged.

**Proof obligation:** Apply Proposition 1 with one summand and Proposition 2's feasibility condition.

**Empirical distinction:** This decoder identity does not guarantee that a jointly trained ARC network matches a separately trained step head. The preregistered native-rate non-inferiority test addresses that learned-policy question.

## Confirmatory Empirical Claim Ledger

| ID | Claim allowed only if | Evidence artifact | Failure response |
|---|---|---|---|
| E1 | ARC+TAC at (k=2) beats post-hoc TAC, cubic spline, and B-spline on Push-T and pooled non-void manipulation under Holm correction | `analyze_arc_tacfold.py` primary table plus validated JSON/checkpoint hashes | Do not state the central positive empirical thesis; document amendment or narrow the paper |
| E2 | ARC+TAC beats ARC+ZOH on at least one non-void (k=2) or (k=4) cell after Holm correction | Mechanism table | Do not attribute gains to TAC decoding |
| E3 | The 90% paired-bootstrap lower bound for ARC versus step native at (k=1) exceeds -5 percentage points | Native-rate no-harm table | State a native-rate tradeoff; the configured main claim fails |
| E4 | All task/rate budgets, protocols, shared-checkpoint hashes, and paired arrays pass analyzer checks | Provenance validation in `load_results` | Exclude invalid cells and rerun; never impute |
| E5 | RGB-plus-proprioception checkpoint and 15 paired CloseSingleDoor outcomes exist and validate | Visual result JSON and checkpoint hash | Remove “visual-policy validation”; retain state-policy scope |

## Strongest Global Counter-Arguments

| Objection | Risk | Response strategy |
|---|---:|---|
| CAT already solves multi-frequency representation more generally | High | Lead with CAT; narrow novelty to the conservative target/decoder contract; avoid priority language |
| Full Spline Policy/B-Spline Policy models were not reproduced | High | Label implemented controls accurately; report this as a limitation; avoid model-family superiority claims |
| Positive simulator results may be seed- or task-specific | Medium | Paired episodes, fixed budgets, Holm correction, complete cell reporting, checkpoint hashes, and no post-hoc task removal |
| Conservation may be irrelevant to feedback control | Medium | Require ARC+TAC versus ARC+ZOH mechanism evidence; discuss invariant as diagnostic, not sufficient condition |
| Visual evidence is underpowered and not language-conditioned | High | Report separately as a confirmation only; never call it a VLA result |
| Bounded clipping changes the learned command | Medium | Define the commanded displacement after standard actuator-bound enforcement; report clipping/saturation incidence |
| Pooled analysis masks heterogeneity | Medium | Publish task-level tables beside pooled inference and identify contrary cells |
| No real robot or wall-clock evaluation | High | Explicitly exclude speed, latency, safety, and hardware generalization claims |

## Logical Flow

```text
Introduction
  action-interface problem
  -> semantic boundary (resolution != latency/speed)
  -> CAT/spline prior art prevents broad novelty
  -> narrow coupled gap and research question

Related Work
  chunking and generative heads
  -> frequency-aware/continuous representations
  -> execution-time methods
  -> required matched controls and unresolved conservation gap

Method
  bounded ARC mean and block-integral target
  -> cumulative TAC reconstruction
  -> feasible saturation repair
  -> conservation, feasibility, and k=1 identity propositions

Experimental Design
  matched factorial arms
  -> paired benchmarks and fixed budgets
  -> six primary tests + Holm
  -> mechanism, no-harm, and visual gates

Results
  provenance audit
  -> primary comparisons
  -> full benchmark cells
  -> mechanism/no-harm
  -> visual confirmation and failure cases

Discussion
  interpret only passed gates
  -> contrast conservation with smoothness and CAT
  -> practical interface impact
  -> simulator, comparator, modality, and speed limitations

Conclusion
  restate only the verified scoped thesis
```

## Argument Strength Assessment

| Sub-argument | Evidence strength now | Logic validity | Counter-argument risk | Required upgrade |
|---|---|---|---|---|
| 1. Problem and novelty boundary | Strong | Valid | Medium | Preserve explicit CAT/spline precedence |
| 2. ARC target semantics | Strong for construction; adequate for learned effect | Qualified | Medium | Complete factorial closed-loop evidence |
| 3. TAC formal guarantees | Strong | Valid under stated assumptions | Medium | Report numerical residual and saturation incidence |
| 4. Causal experimental separation | Strong design; no outcome yet | Valid if protocol passes | High | Complete campaign and provenance audit |
| 5. Scoped practical impact | Weak until results complete | Conditional | High | Pass E1--E5; otherwise narrow or remove |

The draft must not begin a final Results, Abstract, Discussion, or Conclusion claim while Sub-Argument 5 remains weak. Method, theory, background, and protocol prose may be drafted with empirical placeholders, but all outcome language is generated from the locked analyzer output.

## Notes for the Draft Writer

- Use “action resolution” for the intervention; reserve “frequency” for cited terminology or an explicitly defined reciprocal interpretation.
- Call the implemented baselines “matched raw cubic-spline reconstruction” and “matched bounded-error B-spline reconstruction,” not full Spline Policy or B-Spline Policy reproductions.
- Describe CAT as the nearest representation predecessor in the Introduction and Related Work, not only in limitations.
- Use “visual policy” or “RGB-plus-proprioception validation,” never “evaluated VLA.”
- Separate formal guarantees from task-performance evidence in every section.
- State feasibility before exact bounded conservation: the implementation guarantees it by clipping predicted block means before forming sums.
- Report all cells and denominators. Never suppress negative cells to maintain the positive framing.
- Phrase causal interpretation at the level of the matched backbone and simulator protocol.
- Do not use “faster,” “real-time,” “universal,” “first multi-rate,” or “state of the art” unless a separately defined and supported result is added.
- Replace every empirical placeholder only from analyzer-validated artifacts; no inferred or manually transcribed success rates.

## Phase 3 Quality-Gate Status

- Central thesis is clear, specific, and conditional on available evidence: **pass**.
- Five sub-arguments support the thesis: **pass**.
- Every sub-argument has evidence, reasoning, a strong counter-argument, and a response: **pass**.
- Formal propositions state assumptions and distinguish mathematical from empirical claims: **pass**.
- Logical flow covers all paper sections: **pass**.
- Weak empirical impact argument is explicitly gated rather than drafted as fact: **pass**.
- No broad novelty, VLA, wall-clock acceleration, or universal-superiority claim: **pass**.

