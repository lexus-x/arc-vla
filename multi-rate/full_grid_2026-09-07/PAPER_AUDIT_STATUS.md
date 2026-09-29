# Paper correctness audit — live status

Date: 2026-09-23

## Verified today

- `assemble_paper_table.py` currently reports Diffusion Policy PickCube 1X Base as **90%**, not the previously broken 1%.
- The gripper-transition root-cause fix is present: it records the first index of the final constant run and the actual ending command value.
- Focused gripper tests: **9 passed**.
- `assemble_paper_table.py`, `resample_math.py`, and `harness.py` compile successfully.
- `assemble_locked_claim_table.py` now builds the paper-facing table only from two explicit n=400 paired closed-loop files and records their SHA-256/protocol metadata.
- The locked table contains only the five arms present in both files; it refused an earlier attempt to include spline/B-spline because those arms were not paired in the PickCube source.

## Not yet cleared for citation

- Any table not regenerated from raw JSON after this audit.
- The legacy output of `assemble_paper_table.py`; its selector mixes development windows, legacy files, and sample sizes and is not publication-safe.
- Gripper-sync closed-loop claims; the corrected implementation has open-loop tests, but the required corrected closed-loop RoboCasa cells are still pending.
- Learned-governor results until every sealed confirmation task finishes and the pre-registered Holm decision is computed.
- "First," "best," "general," or Q1-readiness language.

## Known risks requiring resolution

- `assemble_paper_table.py` chooses the largest result per arm from filename-filtered JSON and does not encode the full protocol in its merge key. Every selected cell needs a source manifest before publication.
- Some table cells mix sample sizes or cannot compute paired deltas because the chosen native and method arrays differ in length.
- The workspace contains many concurrent uncommitted campaign files. Audit changes must remain isolated and must not rewrite active result artifacts.
- The existing `FEEDBACK_2.md` describes the old gripper bug; it is historical review evidence, not the current implementation state.

## Next audit actions

1. Emit a machine-readable source manifest for every displayed table cell: source JSON, SHA-256, protocol, checkpoint hash, task, policy, rate, arm, and sample count.
2. Reject ambiguous duplicates instead of silently selecting by largest sample count.
3. Regenerate all manuscript tables exclusively from that manifest.
4. Recompute paired statistics and Holm families from raw booleans.
5. Reconcile every prose number with one manifest entry; delete untraceable numbers.
