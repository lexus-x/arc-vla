# SDD ledger — plan: PLAN.md

Repository context: the supplied workspace has no `.git` metadata, so worktree creation,
commit ranges, and commit-based review packages are unavailable. Reviews use before/after
file snapshots and unified diffs instead.

No separate specification is referenced by PLAN.md; rulings below are provisional against
PLAN.md as the sole authority.

## Preflight interface scan

| Items | Producer / consumer interface | Finding |
|---|---|---|
| 1 / 3 | downloaded `human_im` HDF5 files -> vision loader | Compatible; loader must discover and validate actual image keys. |
| 1 / 4 | verified datasets -> training runner | Compatible; runner must fail early if camera observations are absent. |
| 1 / 5 | dataset env metadata/states -> native closed-loop eval | Compatible; bridge must create the environment with the dataset cameras. |
| 2 / 3 | visual policy tensor contract -> loader output | Compatible if state and image tensors stay separate rather than flattening pixels. |
| 2 / 4 | visual policy loss -> multimodal windows | Compatible; requires a vision-specific chunk/training path. |
| 2 / 5 | visual policy sample -> live multimodal history | Compatible; checkpoint must preserve architecture metadata and normalizers. |
| 3 / 4 | raw image/state episodes -> windowed training data | Compatible; `object` must be excluded from proprioception. |
| 3 / 5 | live bridge observations -> runner preprocessing | Compatible; stored and live image orientation/key selection must match. |
| 4 / 5 | trained checkpoint -> native k=1 evaluation | Compatible; runner should load the same EMA weights it saves. |
| 1 self | exact URLs, destination, metadata/camera/demo validation | Internally consistent. |
| 2 self | additive ResNet visual variant; retain state-only policy | Internally consistent. |
| 3 self | load image observations, retain non-privileged proprioception | Internally consistent. |
| 4 self | four tasks, prior budget class, no hyperparameter tuning | Internally consistent once the train/eval split is ruled below. |
| 5 self | native k=1, n>=15, four paper comparisons | Internally consistent. |

Ruling: Use 35 training demonstrations and 15 disjoint held-out demonstrations by default —
the files are described as 50-demo datasets and the existing project protocol evaluates from
held-out demo initial states; training on all 50 would leak every available evaluation state —
if the paper used fresh randomized resets instead, this local comparison will be conservative
and not exactly identical to its episode sampling.

Ruling: Implement long-running acquisition, training, and evaluation as explicit resumable
commands, but do not treat code-only verification as experimental completion — Stage 1 requires
real downloaded files, four trained checkpoints, and four n>=15 result records before scientific
claims are allowed — if immediate execution was intended despite the multi-hour runtime, the
implementation will be ready but the experimental results will remain pending.

Baseline verification before implementation: `python -m pytest -q test_gripper_sync.py
test_resample_invariants.py` -> 9 passed in 2.12s.

Task 1: fix round 1/5 (1 addressed, 0 open — verify-only partial finalization;
snapshot review, no commits available).

Task 1: complete (snapshot review clean; 11 focused acquisition tests and 19 full tests passed).

Task 2: fix round 1/5 (1 addressed, 0 open — campaign artifact compatibility and
reuse validation; snapshot review, no commits available).

Task 2: complete (snapshot review clean; 27 focused Stage 1 tests and 47 full tests passed).
