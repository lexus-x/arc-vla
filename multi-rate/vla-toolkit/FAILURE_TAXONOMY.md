# Failure-mode taxonomy — candidate D (2026-09-04)

Labeled by direct visual review (Claude, multimodal) of near-end keyframes
for all 42 failures from `collect_failures.py`'s 100-episode run on frozen
libero_10 (10 tasks x 10 episodes). Disclosed methodological choice: Claude
itself serves as the MLLM labeler, not an external API — reviewed
representative keyframes per failure, prioritizing the highest-failure-count
tasks first.

## The 10 libero_10 tasks
| id | description | failures/10 |
|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 8 |
| 1 | put both the cream cheese box and the butter in the basket | 2 |
| 2 | turn on the stove and put the moka pot on it | 0 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 0 |
| 4 | put the white mug on the left plate and put the yellow/white mug on the right plate | 2 |
| 5 | pick up the book and place it in the back compartment of the caddy | 2 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 5 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 8 |
| 8 | put both moka pots on the stove | 8 |
| 9 | put the yellow and white mug in the microwave and close it | 7 |

## Failure modes identified (visual review of near-end keyframes)

**Mode 1 — "never completes first grasp" (dominant, most systematic):**
Tasks 0, 1, 7 — ALL share the identical "put both ITEM1 and ITEM2 in the
basket" template with small grocery items (soup can, sauce bottle, cheese
box, butter). In every reviewed failure (16/18 directly inspected), the
basket is completely empty at episode end and the objects remain scattered
in roughly their original positions — the policy never completes even the
FIRST pick-and-place, let alone the second. This is the cleanest, most
homogeneous, most complete failure signature of the whole set.
18/42 failures (43%), 3/3 tasks sharing this exact template.

**Mode 2 — "completes first sub-goal, stalls on second":** Task 8 (moka
pots) — several failures show ONE pot already correctly placed on the stove
and the second still on the table, gripper hovering near the first. A
partial-success/sequencing failure, structurally different from Mode 1 (the
policy CAN complete this pick-and-place type, just not twice in one
episode). Other task-8 failures show neither pot placed (closer to Mode 1) —
this task is a mix.

**Mode 3 — "near-miss / imprecise final placement":** Tasks 4, 6 — gripper
is actively holding the target object (mug) close to the correct target zone
(plate/saucer) at episode end, but placement isn't confirmed complete.
Plausibly a timing issue (520-step budget exhausted mid-multi-step-task)
rather than a grasp/manipulation capability failure — a different root cause
than Modes 1-2, not targeted by this candidate's curation.

**Task 9** (mug + microwave) is mixed between Mode 1 (mug never touched) and
a partial-progress variant (mug picked up, fails at the microwave
place-and-close sub-step) — the most behaviorally complex task (requires
articulated appliance interaction), not cleanly one mode.

## Curation target
Mode 1 (tasks 0, 1, 7) chosen as the primary curation lever:
structurally homogeneous (identical task template), the largest single
cluster, and the cleanest failure signature (total failure, not partial
progress) — the most actionable target for "oversample training data for
the specific skill/task-template the policy demonstrably can't do."

## Implementation
`continued_sft_curated.py`: short continued-SFT (400 steps, batch 32,
matched between arms) from the existing libero_10 checkpoint.
- `--arm targeted`: frames from tasks {0,1,7} get sampling weight 3.0.
- `--arm uniform`: all frames weight 1.0 (matched-compute control).
2 seeds/arm, launched 2026-09-04. Next: re-evaluate all 4 resulting
checkpoints on the full 10-task grid, stats via two-proportion z-test +
paired McNemar (reusing pgad_stats.py).
