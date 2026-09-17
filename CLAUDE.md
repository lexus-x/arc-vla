# Multi-rate VLA action resampling

Research on running vision-language-action policies (SmolVLA, DeepONet-v2, flow-matching) at
control rates other than the one they were trained at — post-hoc resamplers (cubic spline,
B-spline/PCHIP, TAC-Fold/operator-folding, QP-constrained resampling) vs. learned rate
conditioning (ARC) vs. retraining from scratch at the target rate.

## Environments

- `vla_smolvla_libero` (conda) — Python 3.12, torch 2.10+cu128, lerobot 0.5.1 editable at
  `/home/user/lerobot`, robosuite 1.4.0, `hf_libero`. LIBERO / ManiSkill / RoboMimic work.
- `robocasa_uv` (separate env) — RoboCasa needs `numpy<1.24`, which conflicts with the Blackwell
  torch stack above. It **cannot share a process** with `vla_smolvla_libero`; RoboCasa runs
  behind a socket bridge (`robocasa_bridge.py`, port 8765) started in its own env.

## Where things live

- Main eval harness: `multi-rate/full_grid_2026-09-07/harness.py` —
  `python harness.py TASK --k N --arms zoh,spline,tac_fold,... --seed N`. Results land in
  `result_{tag}.json` with per-episode success booleans under `["success"][arm]`, so every
  table is re-derivable from the JSONs. Resamplers live in `resample_math.py` /
  `resample_qp.py` / `resample_bspline2.py`.
- `multi-rate/full_grid_2026-09-07/table_final.txt` and `RESULTS_QP_DRAFT.md` — the most
  current closed-/open-loop tables with McNemar p-values.
- `BENCHMARK_TRACKER.md` (repo root) — converted from the old `.xlsx` scoreboard; frozen
  snapshot as of 2026-09-05, superseded by `full_grid_2026-09-07/` for anything after.
- `multi-rate/vla-vault/` — the older campaign (self-audits, provenance checks, dead-ends
  archive under `output/archive/`). Read `NUMBERS.md` and any `*_AUDIT_*.md` there before
  citing a number from that tree.

## House rules

- Exact McNemar for paired comparisons, Holm correction across a family of tests.
  Pre-register (`PREREG*.md`) before launching a campaign, not after.
- Never write `p < 0.000001` — treat anything past that as a formatting artifact, not evidence.
- A result is "real" only once it clears its pre-registered threshold on the pre-registered
  n; a raw win/loss without that context is not citable.
- State whether a number is open-loop (paired replay) or closed-loop (live rollout) — they are
  not comparable, and a method can win one while losing the other.

## Working conventions

- Working documents are Markdown. Binaries (`.pptx`/`.docx`/`.pdf`/`.xlsx`) are for finished
  external deliverables only — never the live source of a number or a scoreboard.
- Before repeating a research subagent's claim (a number, a "this file shows X"), open the
  file yourself and check it. Subagents summarize; they don't always get it right.
- For anything non-trivial, ask before proceeding rather than guessing at scope.
