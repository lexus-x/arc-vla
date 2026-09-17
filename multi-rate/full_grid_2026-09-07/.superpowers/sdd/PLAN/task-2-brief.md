# Task 2 — execute migration, build corrected table, and audit preregistrations

Implement PLAN.md steps 4–6 exactly after Task 1 passes review:

1. Run `backfill_provenance.py` once. Verify `_pre_backfill_backup/` contains unmodified copies of every preexisting `result_*.json` before accepting the migration.
2. Run `python3 assemble_paper_table.py > table_paper_format_FIXED.txt`. If it exits 1, report every conflict verbatim in `TABLE_FIX_AUDIT.md` and do not resolve it. If it succeeds, compare the fixed table cell by cell against `table_paper_format.txt` (preferred because present), identifying every changed cell and the mismatched-head file wrongly merged before.
3. Recompute the exact paired McNemar p-value for QP vs `spline_satfix` at DP PickCube k=4 n=400 from `result_dp_PickCube-v1_k4_n400.json`, and assess PREREG.md P1–P5 from corrected data only.
4. Write `TABLE_FIX_AUDIT.md` with exactly these sections: `Root cause`; `Corrected vs. previous`; `Corrected paper-format table`; `Verdict per candidate`; `Follow-up bug, not fixed here`. Include both dp and fm output; candidate PASS/FAIL at 1X/2X/4X for QP-eps.05, QP-eps.10, spline+satfix, TAC-Fold+satfix, and B-spline+satfix against PREREG.md/PREREG_QP_CL.md; say plainly whether any claimed win/loss flips. Root cause must cite current file:line locations. Name the precise `heads.decode()` bspline/native branch but do not alter it.

Global prohibitions: do not modify `heads.py`; do not rerun rollouts; do not touch `dp_min.py`, `fm_min.py`, `resample_math.py`, `resample_qp.py`, or `resample_bspline2.py`; do not change protocols or silently resolve conflicts.

Work in `/home/user/Desktop/multi-rate/full_grid_2026-09-07`. There is no usable Git repository, so do not attempt to commit. Write the full execution/test/self-review report to `.superpowers/sdd/PLAN/task-2-report.md` and return only the concise status contract.
