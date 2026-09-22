# ARC--TAC repository audit

## Verdict

ARC--TAC is defensible as a partially novel method, not as the first multi-rate or
continuously resampleable robot policy. Its publishable unit is the coupling of explicit
action-resolution conditioning, block-integral targets, and bounded reconstruction that
preserves feasible predicted block displacement. The formal contract is implemented and
covered by executable tests. The positive empirical claim remains conditional on the locked
Push-T, RoboMimic, RoboCasa, and visual-policy campaign; no existing synthetic artifact is
admissible evidence.

Current engineering status:

- ARC state and RGB-plus-proprioception paths share one checkpoint across `k = 1, 2, 4`.
- TAC saturation repair is feasible by construction after bounding predicted block means.
- The campaign analyzer rejects incomplete cells, mismatched budgets, arm sets, episode
  counts, protocols, and visual checkpoint hashes.
- The preregistered primary claim requires six positive, non-void comparisons after Holm
  correction, a decoder-mechanism win, native-rate non-inferiority, and valid visual evidence.
- The latest targeted suite passes 54 tests; source and runners compile/parse cleanly.

## Files safe to remove after permission

These six artifacts are synthetic demonstrations with hard-coded or calibrated outcomes.
They are not used by the ARC--TAC campaign and create a serious evidence-contamination risk:

- `/home/user/Desktop/multi-rate/run_closed_loop_tri_video.py`
- `/home/user/Desktop/multi-rate/run_full_tri_policy_10_20_40hz_eval.py`
- `/home/user/Desktop/multi-rate/tri_closed_loop_results.json`
- `/home/user/Desktop/multi-rate/tri_policy_all_tasks_10_20_40hz_results.json`
- `/home/user/Desktop/multi-rate/vla_tri_policy_closed_loop_comparison.mp4`
- `/home/user/Desktop/multi-rate/vla_tri_policy_all_tasks_10_20_40hz.mp4`

The following generated transfer archives are superseded by validated v5 and can be removed
after v5 reaches BW2 and its checksum is verified there:

- `/home/user/Desktop/arc_tacfold_bw2_bundle_20260918.tar.gz` (stale source)
- `/home/user/Desktop/arc_tacfold_bw2_bundle_20260918_v2.tar.gz` (stale source)
- `/home/user/Desktop/arc_tacfold_bw2_bundle_20260918_v3.tar.gz` (incorrect dataset paths)
- `/home/user/Desktop/arc_tacfold_bw2_bundle_20260918_v4.tar.gz` (superseded BW2 runtime paths)

Retain `/home/user/Desktop/arc_tacfold_bw2_bundle_20260918_v5.tar.gz` until the campaign and
result return are complete. No file listed here has been removed.

## Do not remove

- Raw datasets, demonstration caches, checkpoints, result JSONs, preregistrations, and full
  campaign logs are provenance artifacts.
- `run_tri_spline_comparison_video_eval.py` is a real evaluation path and is not part of the
  synthetic group above.
- Existing unrelated modified files and active MG3000 logs belong to the user's concurrent
  work and were not changed or cleaned by this audit.
