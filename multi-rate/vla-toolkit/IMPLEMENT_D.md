# IMPLEMENT — Candidate D: failure-mode-clustering-driven data curation

## Design (locked before writing code, per skill's IMPLEMENT stage)

**Suite:** libero_10 (long-horizon, multi-object tasks) — lowest historical
baseline success rate of the 4 suites (more natural failures per rollout
budget) and plausibly richer/more varied failure modes than short single-step
tasks (spatial/object/goal), which matters for a meaningful taxonomy.

**Failure-mode labeler:** Claude itself (multimodal), not an external MLLM
API — reviewing saved keyframes + task description per failed episode and
assigning a failure-mode category + one-line justification. This is a
disclosed methodological choice (not hidden), consistent with how other
papers use GPT-4o/Gemini for the same role.

**Curation mechanism (the actual "method"):** NOT synthetic data generation
(too heavy an engineering lift for the timeline) — targeted REWEIGHTING/
oversampling of EXISTING LIBERO-10 demonstration data during a short
continued-SFT fine-tune, informed by the failure-mode taxonomy (oversample
demos for the task(s)/skill(s) most implicated in the dominant failure
mode(s)), vs. a UNIFORM-sampling continued-SFT at the exact same compute
budget (matched gradient steps/demo-exposures). This isolates "targeted vs.
uniform" cleanly — the toggle this candidate needs per the skill's IMPLEMENT
exit condition.

**Closed-loop structure:**
1. Collect failures from the frozen baseline checkpoint across all 10 libero_10 tasks.
2. Cluster failures into a failure-mode taxonomy (Claude-labeled).
3. Identify the task(s)/skill(s) most implicated in the dominant mode(s).
4. Continued-SFT: targeted-reweighted arm vs. uniform arm, matched compute.
5. Re-evaluate both arms on a held-out task/seed grid, stats via two-proportion
   z-test + paired McNemar (reusing pgad_stats.py's functions).

**Differentiation to keep sharp in writing:** arXiv:2506.06570 does MLLM
failure-taxonomy clustering but never closes the loop with an actual VLA
retrain — this candidate's whole point is the closed loop + quantitative
gain, on a small (<500M) flow-matching VLA + standard LIBERO, which nobody
has done. RECALL (2606.23617) uses pure statistical uncertainty, no semantic
clustering — different signal type entirely, differentiation already solid.

## Stage 1 (now): collect failures
Run the frozen libero_10 checkpoint across all 10 tasks, save keyframes
(start/25%/50%/75%/near-end) + task description + outcome for every FAILED
episode, until a target failure count is reached. Budget: up to 10
episodes/task (100 total rollouts) or until ~30-40 failures collected,
whichever comes first — enough for a meaningful taxonomy without burning the
whole timeline on data collection alone.
