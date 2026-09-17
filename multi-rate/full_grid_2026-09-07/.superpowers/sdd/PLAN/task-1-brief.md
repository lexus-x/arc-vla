# Task 1 — implement provenance and safe table assembly

Implement PLAN.md steps 1–3 exactly:

1. In `harness.py`, add `"head": args.head`, `"seed": args.seed`, and `"fold": args.fold` to the output `res` dict beside `policy`/`k`/`task`.
2. Add `backfill_provenance.py`. For every `result_*.json`, infer provenance from the filename using the harness encoding: `_bspline` means `head=bspline`, `_blocksum` means `head=blocksum`, otherwise `head=step`; `_s<N>` means `seed=N`, otherwise `0`; `_f<N>` means `fold=N`, otherwise `-1`. Before writing anything, copy every input JSON unmodified to `./_pre_backfill_backup/`. Add each inferred field only when absent and never overwrite an existing field. Print how many files were backfilled and head/seed/fold counts. The script must be a one-off that can be run once as required by the plan.
3. In `assemble_paper_table.py`, sort the result glob, skip `n < 10`, skip non-step heads using `d.get('head', 'step')`, retain largest-n selection, and add the exact conflict guard: when an existing and new list have the same length but success rates differ by more than 10 percentage points, collect `(key, arm, file_a, rate_a, file_b, rate_b)`. After all files are processed, print every conflict and exit 1. Do not silently choose among conflicts.

Global prohibitions: do not modify `heads.decode()`, do not rerun rollouts, and do not touch `dp_min.py`, `fm_min.py`, `resample_math.py`, `resample_qp.py`, or `resample_bspline2.py`.

Work in `/home/user/Desktop/multi-rate/full_grid_2026-09-07`. There is no usable Git repository, so do not attempt to commit. Write the full implementation/test/self-review report to `.superpowers/sdd/PLAN/task-1-report.md` and return only the concise status contract.

