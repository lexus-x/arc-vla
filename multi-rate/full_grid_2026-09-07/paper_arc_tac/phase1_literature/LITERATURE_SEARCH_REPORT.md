# Literature Search Report

## Search Strategy

### Scope and evidence profile

- **Research question:** Does joint action-resolution conditioning plus conservative reconstruction reduce multi-rate execution degradation relative to native action heads and post-hoc interpolation?
- **Discipline:** robot learning, imitation learning, visuomotor control, and numerical trajectory reconstruction.
- **Domain evidence profile:** `cs_ml`. Peer-reviewed conference papers and archival preprints were admitted, subject to the same relevance, methodological, and provenance checks.
- **Search date:** 2026-09-17.
- **Date range:** 2016--2026, with older seminal numerical and movement-representation work retained where it directly grounds interpolation or temporal scaling.
- **Language:** English.
- **Primary-source rule:** only original papers, proceedings pages, official project pages, and publisher records were retained. Blogs, surveys, paper-note sites, and repository summaries were used only to locate originals and were not included as evidence.
- **Semantic boundary:** “multi-rate” below means changing the resolution of predicted action blocks and reconstructing controller-rate commands. It does not imply reduced wall-clock completion time. RTC, REMAC, PACE, B-spline Policy, and similar work that explicitly targets latency or speed are therefore adjacent rather than interchangeable.

### Concepts and search strings

Core concepts were: (1) action chunking and temporal action representations, (2) control-rate or action-resolution conditioning, (3) continuous/spline trajectory parameterization, and (4) conservative or constraint-preserving reconstruction.

Representative Boolean strings:

1. `("robot policy" OR "visuomotor policy") AND ("action chunking" OR "action representation") AND (frequency OR rate OR resolution)`
2. `("control frequency" OR "varying control rates" OR "temporal resampling") AND (robot manipulation OR imitation learning)`
3. `(spline OR "continuous action representation" OR trajectory-token) AND (robot policy OR VLA)`
4. `(conservative OR displacement-preserving OR integral-preserving OR saturation) AND (action reconstruction OR trajectory interpolation)`
5. `(Push-T OR RoboMimic OR RoboCasa) AND (diffusion policy OR action chunking OR action representation)`
6. `(RTC OR REMAC OR PACE OR "Temporal Action Selection") AND "action chunking"`

### Databases and progressive search

- **Layer 1, Boolean search:** arXiv, IEEE Xplore, ACM Digital Library, PMLR, RSS proceedings, and OpenReview. Exact-title queries were run for Diffusion Policy, ACT, RoboMimic, RoboCasa, CAT, Spline Policy, B-spline Policy, PACE, REMAC, FAST, RTC, and FreqPolicy.
- **Layer 2, backward chaining:** references and method descriptions in Diffusion Policy, ACT, CAT, Spline Policy, B-spline Policy, and FAST were used to identify action chunking, DMPs, and classical Akima interpolation.
- **Layer 3, forward/current tracking:** 2025--2026 work on action execution and representations was added, notably RTC, REMAC, PACE, TAS, CAT, Spline Policy, B-spline Policy, and the 2026 analysis of action chunking.
- **Layer 4, semantic search:** natural-language searches targeted “one policy across control frequencies,” “trajectory-level continuous action representation,” “frequency-aware action tokens,” and “constraint-preserving reconstruction.” This surfaced CAT and FAST as the closest representation neighbors and separated them from latency-oriented execution methods.

Search interfaces did not expose stable database-wide hit counts, so no unverifiable total-hit number is asserted. Sixty-three surfaced records were manually deduplicated; 38 titles/abstracts were retained for closer screening, 27 original artifacts were assessed, and 25 sources were included. Exclusions were mainly secondary summaries, duplicate versions, unrelated uses of the acronyms PACE/REMAC, and work concerned only with inference acceleration rather than action-resolution transfer.

### Inclusion and exclusion criteria

Included sources had to contribute directly to at least one of: learned temporal action representation, multi-rate/frequency-aware control, action-chunk execution, visual/language/action policy context, numerical reconstruction, or one of the declared benchmarks. Sources also needed an original artifact and enough methodological detail to identify the actual intervention.

Excluded sources were secondary commentary; papers using “frequency” only for spectral regularization with no temporal-action relevance; locomotion-only adaptive-frequency RL without a transferable manipulation-policy interface; and papers whose acronym matched a query but whose method did not address action representation or execution.

### Search saturation

The search stopped after meeting four criteria: the IMRaD minimum of 20 sources was exceeded; every literature-matrix theme had at least three sources; both foundational and 2025--2026 work were represented; and the final exact-title/semantic round added no new method combining all three proposed elements (resolution condition, block-integral supervision, and exact-displacement bounded reconstruction). This last observation supports a narrow gap, not a universal priority claim.

## Coverage Distribution Advisory

`DISTRIBUTIONAL_SKEW_ADVISORY`

- **Dimension:** time distribution
- **Concentration:** 2023--2026 = 22/25 sources (88%).
- **Advisory:** This is a coverage-distribution signal, not a defect. The action-chunking and VLA literature is recent; Akima interpolation and DMPs supply older foundations.
- **Search response:** no expansion. Older generic spline/control literature would add volume without materially sharpening the action-resolution novelty test.

`DISTRIBUTIONAL_SKEW_ADVISORY`

- **Dimension:** methodological distribution
- **Concentration:** empirical computational/simulation or robot experiments = 22/25 sources (88%).
- **Advisory:** The field is experimentally dominated; only a small subset provides formal representation or interpolation properties.
- **Search response:** retain Akima and DMP foundations and require ARC--TAC's conservation, identity, and saturation-feasibility statements to be proved directly rather than inferred from benchmark results.

No geographic-distribution assessment was made because study location is not a meaningful or consistently reported dimension for these algorithmic benchmarks. No venue-family concentration reached 70%.

## Screening Results

- Initial retrievable candidate records after deduplication: 63
- After title/abstract screening: 38
- After original-artifact assessment: 27
- Final included sources: 25
- Peer-reviewed or `cs_ml` peer-reviewed-equivalent: 25/25
- Published in the last five years: 23/25
- Direct closest-overlap set: CAT, Spline Policy, B-spline Policy, FAST, RTC, PACE, and REMAC

### Direct-overlap finding

The broad claim “a policy representation that works across control frequencies” is already occupied most directly by **CAT**, which encodes fixed-real-time trajectories as continuous latent tokens and adds frequency-aware positional encoding. **Spline Policy** and **B-spline Policy** already claim compact continuous trajectories that can be queried or temporally scaled at different resolutions. **FAST** already addresses high-frequency action redundancy through frequency-space sequence tokenization. ARC--TAC therefore cannot credibly claim the first frequency-aware, temporally scalable, or resolution-independent robot action representation.

The defensible partial novelty is the **coupling** of three narrower choices:

1. an explicit action-resolution condition supplied to one policy;
2. fixed-count targets trained as block means/integrals whose native-time coverage changes with that condition; and
3. a conservative cumulative-path decoder whose reconstructed commands preserve each predicted block displacement exactly, with a projection that enforces command bounds whenever the block sum is feasible.

No included source was found to combine all three. CAT is the nearest conceptual predecessor and must be treated as such, not as a peripheral citation. The novelty claim remains vulnerable unless experiments isolate each component and compare against a strong frequency-aware or continuous-trajectory baseline.

## Annotated Bibliography

### 1. Akima (1970), “A New Method of Interpolation and Smooth Curve Fitting Based on Local Procedures.”

- **Primary source:** *Journal of the ACM*, 17(4), 589--602. [doi:10.1145/321607.321609](https://doi.org/10.1145/321607.321609)
- **Type:** Peer-reviewed journal article; seminal numerical method.
- **Method:** Defines local slope estimates and piecewise cubic interpolation using neighboring secant slopes.
- **Key finding:** Produces a smooth interpolant with local rather than global dependence, reducing undesirable oscillatory influence from distant points.
- **Relevance:** TAC-Fold uses an Akima-style cumulative-path interpolation before differencing. This source supports the interpolation ancestry, but not ARC--TAC's learned targets, blockwise conservation claim, or saturation projection.
- **Stance:** Neutral/foundational.
- **Quality:** High for numerical provenance; it contains no robot-learning evaluation.
- **Potential use:** Method provenance and differentiation from generic cubic splines.

### 2. Ijspeert et al. (2013), “Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors.”

- **Primary source:** *Neural Computation*, 25(2), 328--373. [doi:10.1162/NECO_a_00393](https://doi.org/10.1162/NECO_a_00393)
- **Type:** Peer-reviewed journal article; seminal robot movement representation.
- **Method:** Represents discrete and rhythmic motions as stable dynamical systems with learnable forcing terms and temporal scaling.
- **Key finding:** DMPs provide reusable movement representations with attractor stability and modulation of duration and goal.
- **Relevance:** Establishes that temporal scaling of robot trajectories is longstanding. It directly blocks any broad “first temporally scalable action representation” claim.
- **Stance:** Opposes broad novelty; neutral toward the narrow coupled claim.
- **Quality:** High; mature theoretical and empirical foundation.
- **Potential use:** Historical positioning and scope control.

### 3. Mandlekar et al. (2021), “What Matters in Learning from Offline Human Demonstrations for Robot Manipulation.”

- **Primary source:** CoRL 2021, PMLR 164. [Proceedings paper](https://proceedings.mlr.press/v164/mandlekar22a.html), [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)
- **Type:** Peer-reviewed conference paper and benchmark study.
- **Method:** Compares six offline learning methods over simulated and real manipulation datasets of varying quality and introduces the RoboMimic framework/datasets.
- **Key finding:** Demonstration quality, observation modality, stopping criteria, and algorithmic choices materially affect manipulation success.
- **Relevance:** Grounds Lift, Can, and Square as established benchmark cells and motivates matched training/evaluation controls.
- **Stance:** Neutral/benchmark.
- **Quality:** High; broad controlled study with public datasets and code.
- **Potential use:** Experimental protocol and benchmark description.

### 4. Brohan et al. (2022), “RT-1: Robotics Transformer for Real-World Control at Scale.”

- **Primary source:** [arXiv:2212.06817](https://arxiv.org/abs/2212.06817), official project artifact.
- **Type:** Archival preprint with large-scale real-robot experiments.
- **Method:** Tokenizes images, natural-language instructions, and actions in a Transformer trained across a large real-robot dataset.
- **Key finding:** Data and model scaling improve multi-task generalization and robustness in real-world control.
- **Relevance:** Provides early generalist visual-language robot-policy context, but its action interface is not a conservative multi-resolution decoder.
- **Stance:** Neutral/contextual.
- **Quality:** High empirical scale; action-resolution transfer is not isolated.
- **Potential use:** VLA/generalist-policy background.

### 5. Chi et al. (2023), “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.”

- **Primary source:** RSS 2023. [doi:10.15607/RSS.2023.XIX.026](https://doi.org/10.15607/RSS.2023.XIX.026), [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- **Type:** Peer-reviewed conference paper.
- **Method:** Models multimodal action sequences as conditional denoising diffusion and executes them through receding-horizon control.
- **Key finding:** Action diffusion is competitive across 12 manipulation tasks; the paper popularized the Push-T task and action-chunk diffusion baseline used here.
- **Relevance:** ARC--TAC changes the supervised action representation and decoder while retaining this backbone family. Matched-backbone comparisons are therefore essential.
- **Stance:** Supports the backbone choice; neutral on ARC--TAC novelty.
- **Quality:** High; established benchmark and public implementation.
- **Potential use:** Backbone, Push-T protocol, and action-chunk baseline.

### 6. Zhao et al. (2023), “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.”

- **Primary source:** RSS 2023. [Proceedings PDF](https://www.roboticsproceedings.org/rss19/p016.pdf), [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- **Type:** Peer-reviewed conference paper.
- **Method:** Action Chunking with Transformers predicts multi-step action sequences and combines overlapping predictions through temporal ensembling.
- **Key finding:** Chunk prediction reduces effective horizon and enables difficult bimanual manipulation from limited demonstrations.
- **Relevance:** Establishes action chunking and overlap-based temporal aggregation as strong prior art. ARC's block targets are not action chunking per se; the distinction must be explicit.
- **Stance:** Opposes broad action-chunk novelty; supports studying temporal representations.
- **Quality:** High; real-robot evidence with a widely used method.
- **Potential use:** Related work and baseline motivation.

### 7. Brohan et al. (2023), “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.”

- **Primary source:** CoRL 2023, PMLR 229. [Proceedings PDF](https://proceedings.mlr.press/v229/zitkovich23a/zitkovich23a.pdf), [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
- **Type:** Peer-reviewed conference paper.
- **Method:** Co-fine-tunes vision-language models on web-scale vision-language tasks and robot trajectories, expressing actions as text tokens.
- **Key finding:** Web pretraining transfers semantic knowledge and improves generalization in robot control.
- **Relevance:** Defines the modern VLA framing. Because the planned ARC--TAC experiments lack language inputs, they support a visual-action method, not a language-conditioned VLA result.
- **Stance:** Opposes misuse of the VLA label; neutral on the action-resolution method.
- **Quality:** High empirical scale.
- **Potential use:** Terminology and claim boundary.

### 8. Mandlekar et al. (2023), “MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations.”

- **Primary source:** CoRL 2023, PMLR 229. [Proceedings PDF](https://proceedings.mlr.press/v229/mandlekar23a/mandlekar23a.pdf), [arXiv:2310.17596](https://arxiv.org/abs/2310.17596)
- **Type:** Peer-reviewed conference paper.
- **Method:** Transforms a small number of human demonstrations into large synthetic datasets across varied object poses, scenes, and robots.
- **Key finding:** Automatically generated demonstrations can train effective policies on long-horizon and precision tasks.
- **Relevance:** Supports scalable simulation data and the RoboMimic/RoboCasa experimental ecosystem, but does not address action resolution.
- **Stance:** Neutral/benchmark context.
- **Quality:** High; extensive task and data-scale evaluation.
- **Potential use:** Dataset provenance and simulation-method justification.

### 9. Open X-Embodiment Collaboration et al. (2023), “Open X-Embodiment: Robotic Learning Datasets and RT-X Models.”

- **Primary source:** ICRA 2024. [arXiv:2310.08864](https://arxiv.org/abs/2310.08864), [official project](https://robotics-transformer-x.github.io/)
- **Type:** Peer-reviewed conference paper and dataset/model release.
- **Method:** Standardizes data across 22 robot embodiments and trains cross-embodiment RT-X policies.
- **Key finding:** Diverse cross-robot training yields positive transfer on several embodiments.
- **Relevance:** Demonstrates the practical importance of heterogeneous action interfaces and rates. It does not provide an exact-displacement resolution decoder.
- **Stance:** Supports the impact motivation; neutral on novelty.
- **Quality:** High-scale collaborative evidence, though heterogeneous datasets complicate controlled causal attribution.
- **Potential use:** Motivation for reusable action interfaces.

### 10. Nasiriany et al. (2024), “RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots.”

- **Primary source:** RSS 2024. [Official paper](https://robocasa.ai/assets/robocasa_rss24.pdf), [arXiv:2406.02523](https://arxiv.org/abs/2406.02523)
- **Type:** Peer-reviewed conference paper and benchmark release.
- **Method:** Provides diverse kitchen scenes, assets, tasks, demonstrations, and automated data generation for generalist manipulation.
- **Key finding:** Increased synthetic training scale improves simulation and sim-to-real policy performance.
- **Relevance:** Grounds the declared RoboCasa state-policy cells and the separate RGB-plus-proprioception confirmation.
- **Stance:** Neutral/benchmark.
- **Quality:** High; rich open simulation framework, with simulator version sensitivity requiring exact reporting.
- **Potential use:** Benchmark and observation-modality description.

### 11. Prasad et al. (2024), “Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation.”

- **Primary source:** RSS 2024. [Official project and citation](https://consistency-policy.github.io/)
- **Type:** Peer-reviewed conference paper.
- **Method:** Distills a diffusion policy into a consistency model for one/few-step action generation.
- **Key finding:** Reduces inference cost by roughly an order of magnitude while retaining competitive task success on simulation and real robots.
- **Relevance:** Separates model-inference acceleration from action-resolution reconstruction. ARC--TAC must not interpret simulator resolution changes as wall-clock speedups.
- **Stance:** Opposes conflation of action resolution and inference speed.
- **Quality:** High; matched comparisons across RoboMimic, Push-T, and real tasks.
- **Potential use:** Semantic boundary and efficiency-related discussion.

### 12. Octo Model Team et al. (2024), “Octo: An Open-Source Generalist Robot Policy.”

- **Primary source:** [arXiv:2405.12213](https://arxiv.org/abs/2405.12213), [official project](https://octo-models.github.io/)
- **Type:** Archival preprint and open model release.
- **Method:** Trains a Transformer diffusion policy on 800,000 Open X-Embodiment trajectories with flexible task and observation interfaces.
- **Key finding:** The model can be fine-tuned across new robot platforms and action spaces.
- **Relevance:** Shows backbone/interface flexibility but not explicit action-resolution conditioning or conservative reconstruction.
- **Stance:** Supports backbone-agnostic impact; neutral on novelty.
- **Quality:** High-scale empirical evidence and open artifacts.
- **Potential use:** Generalist-policy context and future transfer scope.

### 13. Kim et al. (2024), “OpenVLA: An Open-Source Vision-Language-Action Model.”

- **Primary source:** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246), [official project](https://openvla.github.io/)
- **Type:** Archival preprint and open model release.
- **Method:** Fine-tunes a 7B vision-language backbone to predict discretized robot actions using 970,000 demonstrations.
- **Key finding:** Strong generalist manipulation and efficient downstream adaptation are possible with an open VLA.
- **Relevance:** Provides a credible future backbone for testing whether ARC--TAC transfers beyond a small diffusion policy. It does not itself establish multi-resolution decoding.
- **Stance:** Neutral/contextual.
- **Quality:** High-scale and reproducible relative to closed VLAs.
- **Potential use:** VLA action-head context and limitations.

### 14. Black et al. (2024/2026), “$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control.”

- **Primary source:** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164), [official PDF](https://www.pi.website/download/pi0.pdf)
- **Type:** Archival preprint with large-scale real-robot experiments.
- **Method:** Adds a continuous flow-matching action expert to a pretrained vision-language model and predicts action chunks for dexterous control.
- **Key finding:** Continuous generative action heads scale to diverse robots and complex manipulation.
- **Relevance:** Establishes a modern continuous VLA action head and motivates architecture-agnostic decoding. It also makes RTC a particularly relevant execution comparator.
- **Stance:** Neutral/adjacent.
- **Quality:** High empirical breadth; full training data are not entirely public.
- **Potential use:** Continuous VLA/action-expert context.

### 15. Pertsch et al. (2025), “FAST: Efficient Action Tokenization for Vision-Language-Action Models.”

- **Primary source:** [arXiv:2501.09747](https://arxiv.org/abs/2501.09747), [official PDF](https://www.physicalintelligence.company/download/fast.pdf)
- **Type:** Archival preprint and released tokenizer.
- **Method:** Applies discrete cosine transforms, quantization, and byte-pair encoding to compress high-frequency action sequences; FAST+ is trained across one million trajectories.
- **Key finding:** Frequency-space sequence tokenization improves autoregressive VLA training and handles high-frequency dexterous data better than per-timestep binning.
- **Relevance:** Strong action-representation prior art. FAST changes tokenization/compression, whereas ARC changes supervised temporal aggregation and TAC changes bounded reconstruction. ARC--TAC cannot claim the first action representation addressing high-frequency redundancy.
- **Stance:** Opposes broad novelty; neutral toward the exact coupled claim.
- **Quality:** High empirical relevance with released artifacts.
- **Potential use:** Closest action-tokenization comparison and novelty boundary.

### 16. Black, Galliker, and Levine (2025), “Real-Time Execution of Action Chunking Flow Policies.”

- **Primary source:** NeurIPS 2025. [arXiv:2506.07339](https://arxiv.org/abs/2506.07339)
- **Type:** Peer-reviewed conference paper.
- **Method:** RTC asynchronously generates a new action chunk while executing the current chunk, freezing committed actions and inpainting the remaining overlap.
- **Key finding:** Improves smoothness and robustness under inference delay on dynamic simulation and real bimanual tasks.
- **Relevance:** The intervention is asynchronous latency handling, not resolution-conditioned block-integral learning. It is a necessary semantic comparator whenever “rate” or “faster execution” is discussed.
- **Stance:** Opposes wall-clock acceleration claims from the current simulator intervention.
- **Quality:** High; direct real-time system evaluation.
- **Potential use:** Related work and semantic boundary.

### 17. Su et al. (2025), “FreqPolicy: Efficient Flow-Based Visuomotor Policy via Frequency Consistency.”

- **Primary source:** NeurIPS 2025. [arXiv:2506.08822](https://arxiv.org/abs/2506.08822)
- **Type:** Peer-reviewed conference paper.
- **Method:** Adds frequency-domain consistency constraints to flow-based policies for temporally coherent one-step action generation.
- **Key finding:** Improves fast action generation across simulation, VLA, and real-robot settings, reporting high inference frequency.
- **Relevance:** Uses “frequency” spectrally and computationally rather than as explicit controller action resolution. It is adjacent, and careless terminology would create false overlap.
- **Stance:** Neutral when terminology is precise; opposes generic “frequency-aware policy” novelty.
- **Quality:** High benchmark breadth; emphasis is inference efficiency rather than exact action reconstruction.
- **Potential use:** Related work and terminology disambiguation.

### 18. Weng et al. (2025/2026), “Temporal Action Selection for Action Chunking.”

- **Primary source:** [arXiv:2511.04421](https://arxiv.org/abs/2511.04421)
- **Type:** Archival preprint with simulation and physical-robot experiments.
- **Method:** Caches chunks predicted at different observation times and learns a lightweight selector to choose actions dynamically.
- **Key finding:** Improves the trade-off between reactivity, consistency, and motion coherence.
- **Relevance:** Operates over competing cached chunks rather than learning different action resolutions or conserving block displacement.
- **Stance:** Neutral/adjacent execution work.
- **Quality:** Medium-high; strong reported gains, but recent and not yet an established benchmark standard.
- **Potential use:** Action-chunk execution taxonomy.

### 19. Yang et al. (2026), “Trajectory-Level Continuous Action Representation for Robotic Manipulation” (CAT).

- **Primary source:** [arXiv:2608.24111](https://arxiv.org/abs/2608.24111)
- **Type:** Archival preprint; closest direct overlap.
- **Method:** Encodes trajectories over a fixed real-time interval into continuous latent tokens and uses frequency-aware positional encoding to share temporal coordinates across control frequencies.
- **Key finding:** Reports improvements across multiple backbones, control frequencies, LIBERO, MimicGen, and real-world tasks.
- **Relevance:** CAT already addresses entanglement between action representation and control frequency. ARC differs only if its explicit resolution condition, block-integral targets, and exact bounded reconstruction are jointly emphasized and ablated.
- **Stance:** Directly opposes any broad first-to-multi-rate claim; leaves room for the narrow coupled contribution.
- **Quality:** High relevance and broad reported evaluation; very recent preprint, so independent replication and detailed matched comparisons are limited.
- **Potential use:** Primary novelty comparator and required related-work anchor.

### 20. Tian et al. (2026), “Spline Policy: A Structured Representation for Robot Policies.”

- **Primary source:** [arXiv:2606.07386](https://arxiv.org/abs/2606.07386)
- **Type:** Archival preprint submitted to IEEE.
- **Method:** Replaces fixed-resolution action chunks with predicted spline parameters; the continuous trajectory can be queried at different resolutions and integrated with vector fields, uncertainty propagation, and classical controllers.
- **Key finding:** Demonstrates compatibility with diffusion, flow, Transformer, and VLA backbones in simulation and real-robot cases.
- **Relevance:** Directly overlaps compact structured trajectories and temporal resampling. Unlike ARC--TAC, it does not center block-integral supervision or blockwise exact-displacement projection, but it likely offers richer geometry.
- **Stance:** Opposes broad structured/resampleable representation novelty.
- **Quality:** High conceptual and experimental breadth; recent preprint.
- **Potential use:** Closest continuous-trajectory baseline and novelty boundary.

### 21. Han et al. (2026), “B-Spline Policy: Accelerating Manipulation Policies via B-Spline Action Representations.”

- **Primary source:** [arXiv:2607.09648](https://arxiv.org/abs/2607.09648)
- **Type:** Archival preprint.
- **Method:** Predicts B-spline knots/control points as a continuous action trajectory that can be temporally scaled and executed at varied rates.
- **Key finding:** Reports reduced completion time while maintaining success across simulation and real-world tasks.
- **Relevance:** Strong direct competitor for temporal rescaling and smooth reconstruction. ARC--TAC's distinction is conservative block displacement and bounded feasibility, not being the first spline-like speed/resolution method.
- **Stance:** Opposes broad temporal-scaling and acceleration novelty.
- **Quality:** High relevance; recent preprint with limited independent validation.
- **Potential use:** Required baseline/related-work comparison and semantic contrast with wall-clock speed.

### 22. Nie et al. (2026), “PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking.”

- **Primary source:** [arXiv:2606.00537](https://arxiv.org/abs/2606.00537)
- **Type:** Archival preprint with large-scale simulation and real-robot experiments.
- **Method:** Selects how much of a predicted chunk to execute by detecting low-speed phase transitions, without retraining the policy.
- **Key finding:** Adaptive execution horizons improve aggregate performance on RoboTwin2.0 and physical ALOHA/Franka tasks.
- **Relevance:** Changes the executed prefix length at test time; ARC changes the supervised action resolution and reconstructs controller commands. The two methods address orthogonal decisions and could be complementary.
- **Stance:** Neutral/adjacent; opposes claims that ARC is the first adaptive temporal execution method.
- **Quality:** High reported evaluation breadth; recent preprint.
- **Potential use:** Action-execution taxonomy and complementarity.

### 23. Wang et al. (2026), “Real-Time Robot Execution with Masked Action Chunking” (REMAC).

- **Primary source:** ICLR 2026. [arXiv:2601.20130](https://arxiv.org/abs/2601.20130), [official project](https://remac-async.github.io/)
- **Type:** Peer-reviewed conference paper.
- **Method:** Fine-tunes a pretrained policy using masked action chunking to correct asynchronous intra-chunk mismatch and uses prefix-preserving sampling for continuity.
- **Key finding:** Improves completion and robustness under varying inference delays without added latency.
- **Relevance:** Addresses execution-time perception/action misalignment rather than resolution-conditioned targets. It is a strong comparator for real-time claims but not for exact block reconstruction.
- **Stance:** Opposes latency/real-time conflation; neutral on the narrow ARC--TAC method.
- **Quality:** High; simulation and real-robot delay studies.
- **Potential use:** Related work and limitation boundary.

### 24. Feng et al. (2026), “Demystifying Action Space Design for Robotic Manipulation Policies.”

- **Primary source:** ICLR 2026. [arXiv:2602.23408](https://arxiv.org/abs/2602.23408), [OpenReview paper](https://openreview.net/pdf/f5c8542e34d2a81d82d1f1a93790f44c368250db.pdf)
- **Type:** Peer-reviewed conference paper.
- **Method:** Systematically varies spatial and temporal action abstractions across more than 500 trained models and 13,000 real-robot rollouts.
- **Key finding:** Action-space design substantially affects learnability and stability; delta actions are generally favorable, while joint- and task-space controls have complementary strengths.
- **Relevance:** Strong evidence that action representation is a substantive method choice, not plumbing. It also raises the bar for ablations and matched control interfaces.
- **Stance:** Supports the significance of the research question; neutral on specific novelty.
- **Quality:** High due to unusually large controlled real-robot study.
- **Potential use:** Motivation, method framing, and ablation rationale.

### 25. Lazzati et al. (2026), “Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?”

- **Primary source:** [arXiv:2608.02547](https://arxiv.org/abs/2608.02547)
- **Type:** Archival preprint with simulation and real-robot experiments.
- **Method:** Tests competing explanations for action chunking and compares against delayed policies and explicit policy ensembles.
- **Key finding:** Attributes gains primarily to non-Markovian expressivity, reduced compounding error, and implicit ensembling rather than common explanations such as simple temporal consistency.
- **Relevance:** Warns against attributing gains from ARC targets to smoothness alone. Mechanism claims need decoder and representation ablations under identical chunk structure.
- **Stance:** Opposes simplistic mechanism narratives; supports rigorous ablation.
- **Quality:** High methodological relevance; very recent preprint.
- **Potential use:** Mechanism interpretation and discussion.

## Literature Matrix

| Source | Action representation | Multi-rate / temporal scaling | Conservative / constrained reconstruction | Execution / latency | VLA / generalist context | Benchmark relevance | Method | Quality |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Akima (1970) | x |  | main |  |  |  | Numerical analysis | High |
| Ijspeert et al. (2013) | main | main | x |  |  |  | Dynamical systems | High |
| Mandlekar et al. (2021) | x |  |  |  |  | main | Empirical benchmark | High |
| Brohan et al. (2022), RT-1 | x |  |  | x | main |  | Transformer policy | High |
| Chi et al. (2023), DP | main | x |  | x | x | main | Diffusion policy | High |
| Zhao et al. (2023), ACT | main | x |  | main | x |  | Transformer/CVAE | High |
| Brohan et al. (2023), RT-2 | x |  |  |  | main |  | VLA | High |
| Mandlekar et al. (2023), MimicGen |  |  |  |  | x | main | Data generation | High |
| Open X-Embodiment (2023) | x | x |  |  | main | x | Dataset/generalist policy | High |
| Nasiriany et al. (2024), RoboCasa |  |  |  |  | x | main | Simulation benchmark | High |
| Prasad et al. (2024) | x |  |  | main | x | main | Consistency distillation | High |
| Octo Team et al. (2024) | main | x |  |  | main | x | Generalist diffusion policy | High |
| Kim et al. (2024), OpenVLA | main |  |  | x | main | x | Autoregressive VLA | High |
| Black et al. (2024), $\pi_0$ | main | x |  | x | main | x | Flow-matching VLA | High |
| Pertsch et al. (2025), FAST | main | x |  | x | main | x | DCT tokenization | High |
| Black et al. (2025), RTC | x | x | x | main | main | x | Async inpainting | High |
| Su et al. (2025), FreqPolicy | main | x |  | main | x | x | Frequency consistency | High |
| Weng et al. (2025), TAS | x | x |  | main |  | x | Learned action selection | Medium-High |
| Yang et al. (2026), CAT | main | main | x | x | x | main | Continuous latent trajectory | High |
| Tian et al. (2026), Spline Policy | main | main | x | x | main | x | Spline/vector field | High |
| Han et al. (2026), BSP | main | main | x | main | x | x | B-spline policy | High |
| Nie et al. (2026), PACE | x | main |  | main | x | x | Test-time horizon selection | High |
| Wang et al. (2026), REMAC | x | x | x | main | x | x | Masked chunk adaptation | High |
| Feng et al. (2026) | main | x |  | x | x | x | Large-scale empirical study | High |
| Lazzati et al. (2026) | main | x |  | main |  | x | Mechanistic experiments | High |

## Identified Gaps

1. **Exact-displacement gap.** Current learned trajectory representations emphasize compactness, smoothness, temporal querying, or frequency invariance. The reviewed papers do not make blockwise equality between predicted displacement and reconstructed controller commands the central invariant, especially after command-bound correction.

2. **Target/decoder coupling gap.** CAT conditions temporal coordinates and learns continuous latent tokens; Spline Policy and BSP learn curve parameters; FAST compresses trajectories. None of the reviewed sources was found to train a fixed number of block-integral targets at explicitly conditioned resolutions and pair them with a conservative cumulative-path decoder.

3. **Causal attribution gap.** Much prior work changes representation, backbone, inference schedule, and execution together. A matched-backbone decomposition of `(step head + post-hoc reconstruction)`, `(resolution-conditioned blocks + ZOH)`, and `(resolution-conditioned blocks + conservative fold)` would isolate learning from decoding more cleanly.

4. **Control-rate semantics gap.** “Frequency,” “rate,” “speed,” and “resolution” are used for different phenomena: spectral content (FAST/FreqPolicy), controller sampling (CAT/SP/BSP), replanning horizon (PACE/TAS), and wall-clock inference/latency (Consistency Policy/RTC/REMAC). ARC--TAC can add value by defining action resolution precisely, but only if it avoids claiming wall-clock acceleration from simulation subsampling.

5. **Evidence gap for impact.** A positive method paper needs more than superiority to ZOH and a native step head. CAT, Spline Policy, or B-spline Policy is the meaningful modern competitor family. If implementation constraints prevent a full matched reproduction, at minimum the paper needs matched spline/continuous-trajectory baselines, strong ablations, and an explicit limitation that no head-to-head CAT comparison was performed.

6. **Generality gap.** The planned Push-T, RoboMimic, and state-based RoboCasa cells establish action-resolution transfer for imitation policies. One RGB-plus-proprioception RoboCasa task tests observation-backbone compatibility, but it does not establish language conditioning, cross-embodiment transfer, or real-time hardware acceleration.

## Novelty and Positioning Verdict

**Verdict: partially novel and potentially publishable, with a narrow claim.** The literature supports a positive engineering method paper if the empirical gates are positive and the method is positioned as a conservative interface between a resolution-conditioned action head and a controller-rate command stream.

The strongest defensible statement is:

> ARC--TAC combines explicit action-resolution conditioning, block-integral supervision, and bounded exact-displacement reconstruction in one learned policy, reducing degradation when the same policy is queried at multiple declared action resolutions.

Statements that are not supported by the literature search include: “first multi-rate robot policy,” “first frequency-aware action representation,” “first continuously resampleable action policy,” “first temporally scalable robot trajectory representation,” or “faster task execution.” CAT, Spline Policy, BSP, DMPs, FAST, RTC, PACE, and REMAC collectively preclude those broader claims.

The key novelty risk is **CAT**. Its frequency-aware shared temporal coordinate system directly addresses cross-frequency representation. Reviewers may regard scalar resolution conditioning plus block averaging as a simpler special case unless TAC's exact-displacement and feasibility guarantees produce a measurable advantage and the component ablations show that post-hoc interpolation alone does not explain the result.

## Recommended Sources by Paper Section

| Section | Key sources |
|---|---|
| Introduction / motivation | Feng et al. (2026); CAT; Diffusion Policy; Open X-Embodiment; FAST |
| Related temporal action representations | ACT; DMPs; CAT; Spline Policy; B-spline Policy; FAST; FreqPolicy; Lazzati et al. |
| Related execution and latency methods | Consistency Policy; RTC; REMAC; PACE; TAS |
| Method provenance | Akima (1970); Diffusion Policy; ACT; CAT; Spline Policy; BSP |
| Experimental benchmarks | Diffusion Policy/Push-T; RoboMimic; MimicGen; RoboCasa |
| Visual/generalist policy context | RT-1; RT-2; Octo; OpenVLA; $\pi_0$; Open X-Embodiment |
| Discussion and limitations | CAT; Spline Policy; BSP; RTC; REMAC; PACE; Lazzati et al. |

## Phase 1 Quality-Gate Summary

- Search strategy, strings, screening rules, and limitations documented: **pass**.
- IMRaD minimum source count (20): **pass, 25 included**.
- Every included source annotated: **pass**.
- Every literature-matrix theme covered by at least three sources: **pass**.
- At least two actionable gaps: **pass, six identified**.
- Peer-reviewed or `cs_ml` admissible archival-source ratio: **pass, 25/25**.
- More than 50% from the last five years: **pass, 23/25**.
- Broad novelty claim: **fail**.
- Narrow coupled novelty claim: **provisionally supported**, contingent on positive matched experiments and comparison against the CAT/spline-policy family.
