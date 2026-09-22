# Paper Configuration Record

| Field | Confirmed configuration |
|---|---|
| Working title | ARC-TAC: Action-Resolution Conditioning with Conservative Folding for Multi-Rate Robot Policies |
| Paper type | Positive-results engineering method paper (IMRaD) |
| Discipline | CS / Engineering / Robotics and Robot Learning |
| Domain Evidence Profile | cs_ml |
| Novelty target | Partial but clear: action-resolution-conditioned block-integral learning coupled to displacement-conserving TAC-Fold decoding |
| Impact target | Decent practical impact: robust deployment of one policy across controller rates without per-rate retraining |
| Central research question | Does joint action-resolution conditioning plus conservative reconstruction reduce multi-rate execution degradation relative to native action heads and post-hoc interpolation? |
| Main benchmarks | Push-T; RoboMimic Lift, Can, Square; RoboCasa four manipulation tasks |
| Visual confirmation | RoboCasa CloseSingleDoor with RGB plus proprioception |
| Evidence | Formal conservation, identity, and saturation-feasibility properties; paired closed-loop experiments; mechanism and ablation tests |
| Claim boundary | Positive method claim; no universal superiority claim; no language-conditioned claim without language input |
| Semantic boundary | The current simulator intervention changes action representation/resolution and reconstructs controller-rate commands; it does not by itself establish wall-clock task acceleration |
| Target venue | Q2 robotics / automation journal, selected after scope-fit check |
| Citation format | IEEE |
| Output | English LaTeX manuscript and BibTeX |
| Target length | Approximately 8,000 words |

Confirmed by the user on 2026-09-17: “positive result, method paper” and “partially novel and decent impact.”
