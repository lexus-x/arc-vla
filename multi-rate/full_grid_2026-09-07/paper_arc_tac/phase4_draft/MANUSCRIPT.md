# ARC--TAC: Action-Resolution Conditioning with Conservative Folding for Multi-Rate Robot Policies

> Evidence-bound pre-results draft. The abstract, numerical Results, Discussion, and Conclusion
> must be completed only from `analyze_arc_tacfold.py` after every provenance check and claim gate
> passes. Bracketed text is deliberately unresolved and is not a result.

## 1. Introduction

The temporal form of a robot policy's output is part of the learned interface, not merely an
implementation detail. Action chunking changes the information available to a policy, its
replanning behavior, and its exposure to compounding error [1], [2]. Large controlled studies
further show that temporal and spatial action abstractions materially affect manipulation-policy
learnability and stability [3]. Yet a policy trained to emit controller-step actions is often
deployed in systems that require a different action resolution. Retraining one model for every
resolution is expensive, while post-hoc interpolation changes the executed trajectory without
giving the learned head an explicit account of that change.

This paper studies **action resolution**: the number `k` of native demonstration actions
represented by one predicted action block. This intervention is distinct from controller clock
rate, asynchronous inference latency, the number of predicted actions executed before replanning,
and wall-clock task acceleration. Methods such as real-time chunking, masked execution, and
phase-aware prefix selection address those adjacent deployment problems [4]--[6]; success under
the present intervention does not establish their speed or latency claims.

Continuous and multi-rate action representations are established prior art. Dynamic movement
primitives provide temporal scaling, FAST encodes action trajectories in a frequency basis, and
CAT learns a shared continuous representation across control frequencies [7]--[9]. Spline Policy
and B-Spline Policy predict continuously queryable trajectory representations [10], [11]. These
works rule out broad claims that the present method is the first frequency-aware, temporally
scalable, or resampleable robot policy. They also sharpen a narrower question: can a compact
action head expose the requested resolution explicitly while maintaining a checkable physical
contract between each learned token and the bounded controller-rate commands reconstructed from
it?

We propose ARC--TAC, a coupled action interface. **Action-Resolution Conditioning (ARC)** trains
one diffusion action head over declared resolutions using block-mean targets. Each continuous
token therefore denotes a block displacement after multiplication by its declared block size.
**Temporal Action Conservation (TAC)** interpolates the cumulative sequence of those block
displacements and differences the refined path into controller-step commands. A bounded repair
preserves each feasible block displacement while enforcing elementwise action limits. ARC--TAC is
intentionally modest: scalar conditioning and block averaging are simple, and cumulative
interpolation is classical. The partially novel contribution is their explicit target--decoder
contract and its matched empirical test against post-hoc reconstruction.

The paper makes three contributions. First, it defines a fixed-count, resolution-conditioned
block-integral target for one policy queried at `k in {1,2,4}`. Second, it provides a conservative
decoder with blockwise displacement conservation, guaranteed feasibility after bounding predicted
means, and native-resolution identity. Third, it preregisters a paired closed-loop evaluation on
Push-T, RoboMimic, and RoboCasa, with matched cubic-spline and bounded-error B-spline
reconstruction controls, component ablations, and an RGB-plus-proprioception confirmation. The
positive empirical claim is made only if all locked gates pass.

## 2. Related Work

### 2.1 Chunked action prediction

Diffusion Policy models multimodal visuomotor behavior through iterative action-sequence
denoising and established Push-T as a common test bed [1]. ACT predicts action chunks to improve
fine-grained manipulation [2]. Later work separates several reasons chunking can help, including
non-Markovian expressivity, reduced compounding error, and implicit ensembling [12]. ARC retains a
chunked diffusion backbone; consequently, comparisons that change chunk structure or backbone
would confound the claimed mechanism. Our experiment instead holds the backbone, demonstrations,
optimization budget, receding-horizon execution length, and paired evaluation conditions fixed.

### 2.2 Multi-rate and continuous representations

Temporal scaling predates modern robot learning: dynamic movement primitives explicitly separate
trajectory shape from temporal evolution [7]. FAST compresses trajectories in a discrete cosine
basis for efficient tokenization [8]. Most directly, CAT learns a trajectory-level continuous
action representation with a shared temporal coordinate system across frequencies [9]. Spline
Policy predicts continuous spline trajectories compatible with several policy families [10], and
B-Spline Policy uses a B-spline action representation for temporal rescaling [11]. ARC--TAC does
not supersede these richer representations. Its narrower design goal is a minimal action-head
interface whose block-displacement meaning survives bounded reconstruction exactly.

The evaluated spline arms are matched reconstruction controls, not full reproductions of Spline
Policy or B-Spline Policy. The cubic arm interpolates cumulative predicted block sums. The
B-spline arm follows an adaptive bounded-error fitting procedure on the same cumulative points.
Calling these controls full policy reproductions would overstate the experiment because the
learned heads and training objectives of the published methods are not reproduced.

### 2.3 Execution-time adaptation

Real-Time Chunking inpaints action chunks to bridge inference delay [4], REMAC trains masked
chunks for asynchronous execution [5], and PACE selects an executed prefix according to motion
phase [6]. These methods change when predictions become available or how much of a chunk is
executed. ARC changes the supervised temporal meaning of the output token, while TAC reconstructs
controller-rate commands from that token. The dimensions are complementary, but the present
study does not measure latency, throughput, or task completion time.

## 3. Method

### 3.1 Problem formulation

Let `a_t in [-1,1]^D` be a native demonstration action. The first `D_c` dimensions are continuous
delta commands. The remaining dimensions are held commands, such as a gripper state, for which
interpolation is inappropriate. A policy receives two observations and predicts `B=8` action
tokens. ARC supports the declared set `K={1,2,4}`. Resolution `k` means that token `b` summarizes
the native indices `bk,...,bk+k-1`; it does not mean that the simulator clock is accelerated.

During closed-loop evaluation, every arm replans after eight controller steps. Thus ARC predicts
eight blocks but executes the first `8/k` reconstructed blocks at resolution `k`, matching the
step-head execution horizon. All compared arms receive the same initial condition and deterministic
diffusion noise indexed only by episode and replan number.

### 3.2 Action-Resolution Conditioning

For continuous coordinate `d`, ARC's target is the block mean

`mu^(k)_{b,d} = (1/k) sum_{j=0}^{k-1} a_{bk+j,d}`.

The associated block displacement is

`S^(k)_{b,d} = k mu^(k)_{b,d}`.

Held dimensions use the first value in the block and are repeated causally during decoding. The
policy is conditioned by the normalized scalar `r(k)=log2(k)/2`. Training examples for all three
resolutions are constructed from each raw action window, and one checkpoint is optimized jointly
over their union. Targets are normalized after construction, rather than with native per-step
action statistics, so the denormalized output retains the intended block-mean scale.

ARC supplies a fixed tensor shape while changing native-time coverage: eight outputs represent
8, 16, or 32 native actions for `k=1,2,4`. It does not assume that the network necessarily uses
the rate condition. That learned effect is tested by ARC+ZOH versus step-head+ZOH; the decoder's
incremental effect is tested by ARC+TAC versus ARC+ZOH.

### 3.3 Temporal Action Conservation

Given predicted means, the decoder first clips each continuous mean to `[-1,1]` and forms
`S_b=k clip(mu_hat_b,-1,1)`. Define cumulative block endpoints by `C_0=0` and
`C_{b+1}=C_b+S_b`. TAC estimates local Akima-style slopes on the cumulative sequence and evaluates
a piecewise cubic Hermite path at `k` equal subintervals per block. Controller-rate commands are
the consecutive differences of the refined cumulative path.

Unconstrained cubic interpolation can overshoot the actuator bounds. TAC's saturation repair
clips each reconstructed block and redistributes the resulting sum deficit over entries that can
still move toward the target. Redistribution repeats until the target sum is reached or all
coordinates are at a bound. Continuous coordinates are repaired independently; held dimensions
remain causal holds.

### 3.4 Formal properties

**Lemma 1 (ARC target identity).** For every training block,
`S^(k)_{b,d}=sum_j a_{bk+j,d}`.

*Proof.* Substitute the definition of the block mean and cancel `k`. This identity concerns the
supervised target; prediction error remains possible.

**Proposition 1 (endpoint conservation).** Suppose a reconstructed cumulative path includes both
endpoints of every block. If controller commands are consecutive path differences, their sum in
block `b` equals `S_b`.

*Proof.* The finite sum telescopes. All interior samples cancel, leaving
`C_{b+1}-C_b=S_b`. The result is independent of the particular interpolant and holds up to
floating-point tolerance in the implementation.

**Proposition 2 (bounded-repair feasibility).** A scalar sequence `v_1,...,v_k in [-1,1]` with
sum `S` exists if and only if `|S|<=k`. ARC's bounded predicted mean guarantees this condition.
The clip-and-redistribute procedure terminates with the requested sum and bounds in exact
arithmetic.

*Proof.* Necessity follows from the triangle inequality. For sufficiency, the constant sequence
`v_j=S/k` is feasible. In the implemented procedure, a redistribution either closes the deficit
or moves at least one additional entry to the relevant bound. At most `k` entries can become
newly saturated, so the process terminates. The vector result follows coordinatewise.

**Proposition 3 (native-resolution identity).** At `k=1`, TAC returns the bounded predicted mean
unchanged.

*Proof.* Each block contains one difference. By Proposition 1 it equals its block sum, and the
feasibility step has no further degree of freedom.

These are interface guarantees, not stability or task-success theorems. Their empirical value
depends on whether conserving the learned displacement improves feedback-control outcomes.

### 3.5 Compared interfaces

The step-head checkpoint is evaluated with native execution, equal-split zero-order hold (ZOH),
raw cumulative cubic spline, raw bounded-error B-spline, and post-hoc saturation-corrected TAC.
The ARC checkpoint is evaluated with ZOH, cubic spline, B-spline, unconstrained TAC, and
saturation-corrected TAC. This factorial structure distinguishes joint resolution learning from
the reconstruction rule while preserving a common backbone and rollout protocol.

## 4. Experimental Design

We use Push-T with 200 demonstrations, 400 paired evaluation episodes, and 30,000 optimization
steps. RoboMimic Lift, Can, and Square each use 200 demonstrations, 100 paired episodes, and
30,000 steps [13]. Four RoboCasa tasks---TurnOffSinkFaucet, CoffeePressButton,
TurnOffMicrowave, and CloseSingleDoor---use 39 demonstrations, 100 deterministic random-reset
episodes, and 15,000 steps [14]. A separate CloseSingleDoor visual policy uses RGB plus
proprioception, a fixed 35/15 demonstration split, and 15,000 steps. It is supporting evidence,
not a language-conditioned VLA evaluation.

The primary outcome is paired closed-loop success. Six primary comparisons test ARC+TAC against
post-hoc TAC, cubic spline, and B-spline at `k=2`, separately for Push-T and manipulation tasks
pooled across non-void paired cells. Each comparison must have a positive success-rate difference,
a non-void discordant set, and a two-sided exact McNemar `p` value below 0.05 after Holm correction.
The mechanism gate requires ARC+TAC to beat ARC+ZOH in at least one declared `k=2` or `k=4` cell
after correction. Native-resolution no-harm pools all eight state tasks and requires the lower
bound of a paired-bootstrap 90% interval to exceed -5 percentage points.

Before analysis, the executable validator requires all 48 state-policy task--rate--head cells,
the exact arm sets and episode counts, fixed budgets and protocols, complete paired arrays, and
shared checkpoint hashes. The visual gate independently requires one checkpoint and complete
paired results at all three declared resolutions. No missing cell is imputed, and no task can be
removed after outcomes are observed.

## 5. Results

`[LOCKED: insert analyzer-generated primary table, all task-level cells, mechanism table,
native-rate interval, visual table, and provenance statement. Do not write narrative outcome
claims unless the corresponding gate is PASS.]`

## 6. Discussion

`[LOCKED: interpret only passed gates. Report null and negative cells, distinguish conservative
reconstruction from generic smoothness, and retain the limitations: simulation only, matched
reconstruction controls rather than full CAT/Spline Policy reproductions, one small visual task,
no language input, no real robot, no latency claim, and no stability theorem.]`

## 7. Conclusion

`[LOCKED: restate only the verified scoped thesis and formal guarantees.]`

## References

[1] C. Chi et al., “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” RSS, 2023.

[2] T. Z. Zhao et al., “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” RSS, 2023.

[3] Feng et al., “Demystifying Action Space Design for Robotic Manipulation Policies,” ICLR, 2026.

[4] K. Black, N. Galliker, and S. Levine, “Real-Time Execution of Action Chunking Flow Policies,” NeurIPS, 2025.

[5] Wang et al., “Real-Time Robot Execution with Masked Action Chunking,” ICLR, 2026.

[6] Nie et al., “PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking,” 2026.

[7] A. J. Ijspeert et al., “Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors,” Neural Computation, 2013.

[8] K. Pertsch et al., “FAST: Efficient Action Tokenization for Vision-Language-Action Models,” 2025.

[9] Yang et al., “Trajectory-Level Continuous Action Representation for Robotic Manipulation,” 2026.

[10] Tian et al., “Spline Policy: A Structured Representation for Robot Policies,” 2026.

[11] Han et al., “B-Spline Policy: Accelerating Manipulation Policies via B-Spline Action Representations,” 2026.

[12] Lazzati et al., “Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?” 2026.

[13] A. Mandlekar et al., “What Matters in Learning from Offline Human Demonstrations for Robot Manipulation,” CoRL, 2021.

[14] S. Nasiriany et al., “RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots,” RSS, 2024.
