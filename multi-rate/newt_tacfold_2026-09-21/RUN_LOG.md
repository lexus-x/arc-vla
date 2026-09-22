# Run log

## Checkpoint calibration

The official `nicklashansen/newt` 20M state checkpoint was evaluated with Newt's actual protocol
(horizon 3, execute the returned MPPI action, replan every step) on 20 fixed tasks and seeds `0..19`.
It scored **297/400 = 74.25%**, so the checkpoint satisfies the requested ~70% base criterion.
Per-task outcomes are in `official_native_validation.json`.

## Rejected adapters

1. Treating the internal horizon-8 MPPI mean as an action chunk scored 136/400 = 34.0%. This is not
   Newt's output contract, so no confirmatory comparison was launched.
2. Paired open-loop replay passed the two-episode engineering check, but its exploratory screen showed
   TAC-Fold did not beat the splines (for example, PickCube-eepose: native 8/10, cubic 0/10,
   B-spline 1/10, TAC-Fold 0/10). Native replay also changed one PullCube outcome at seed 2, violating
   the deterministic-pair assumption. The screen stopped immediately; no confirmatory result is claimed.

## Decision

Do not use Newt as the TAC-Fold paper benchmark. It is a strong single-action receding-horizon policy,
not an action-chunk policy. The repository's existing ManiSkill Diffusion Policy evidence is compatible
with TAC-Fold and shows large margins on some tasks, but no credible pretrained 20-task checkpoint suite
was found. Expanding that suite requires training additional task policies or reducing the requested
suite size; reporting the Newt replay as a positive result would be misleading.
