# Preregistration: Newt Panda20 trajectory-reconstruction benchmark

Frozen before any replay-arm outcomes were inspected. The earlier `PREREG.md` chunk experiment is
superseded because Newt is a one-action receding-horizon controller; its internal MPPI mean is not
an action-chunk output. Its arm outcomes are excluded.

## Question

After 2x temporal decimation of successful and unsuccessful trajectories from the official pretrained
Newt controller, does TAC-Fold preserve ManiSkill3 task success better than cubic spline and smoothing
B-spline reconstruction?

## Protocol

- Suite: the 20 tasks in `suite.json`, fixed before validation.
- Checkpoint: `nicklashansen/newt`, `soup-20M-default.pt`, SHA-256
  `54d76a7dc98d41fc882d446a72892e28e779f7a2f2f17c096ff336f1fe4a2232`.
- Source trajectory: official Newt state-policy evaluation (planning horizon 3, MPPI action, replan every
  environment step), 100 seeds per task (`0..99`).
- Design: paired deterministic open-loop replay from the same reset seed. This is a trajectory
  reconstruction benchmark, not a closed-loop policy comparison.
- Decimation: sum non-overlapping pairs (`k=2`) in the first six delta-pose dimensions; leave an odd
  final step unchanged. Gripper is causal zero-order held within each pair.
- Arms: native replay, equal-split ZOH, cubic spline on cumulative block sums, smoothing B-spline
  (`s=0.01`) on cumulative block sums, and TAC-Fold. No post-hoc arm-specific clipping or tuning.
- Outcome: final-step ManiSkill success. Report every task, including floors and ceilings.

## Statistics

- Primary A: TAC-Fold vs cubic spline, pooled paired exact two-sided McNemar test.
- Primary B: TAC-Fold vs B-spline, same test.
- Holm correction over these two primary tests; alpha 0.05.
- Report micro and macro success, percentage-point deltas, discordant counts, and all per-task rates.
- The prior 20-seed native-only validation (297/400, 74.25%) selected the checkpoint/protocol but did
  not expose any replay-arm outcome. It is not included in the confirmatory sample.
