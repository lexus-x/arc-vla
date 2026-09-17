# Fable review: gripper_sync is invalid, must be fixed before any result is trusted

## The bug

`gripper_sync(chunk, raw_chunk, n_hold, k)` in `resample_math.py` (lines 189-234)
receives the FULL, undecimated `raw_chunk` (every one of harness.py's 8 native
steps) and its transition-walk loop is mathematically equivalent to
`rebuilt = raw.copy()` — verify this yourself by tracing it: each transition sets
every subsequent index to that transition's raw value, telescoping to an exact
copy. Your own test proves it: `test_gripper_sync.py` line 18,
`np.testing.assert_array_equal(synced[:, 2], raw[:, 2])`. And
`eval_gripper_sync_openloop.py` line 73/77 compares gripper_sync's output against
that SAME `raw` array it was given full access to (`target = raw[:, gripper_dim]`).
So the ~97.5-100% "reconstruction accuracy" table in `PROPOSAL.md` is guaranteed
by construction — it proves nothing about reconstruction quality, it's circular.

Every other resampler in this codebase — zoh, spline, tac_fold, pchip, bspline,
qp — is deliberately restricted to `block_sum` (via `coarsen_delta`), which
genuinely discards the intra-block trace. That's not incidental: it's the stated
premise of the whole research program (`RESULTS_QP_DRAFT.md`'s method section:
"Given a policy's k-step-decimated action block sums S_1..S_B ... recover the
fine-rate sequence"). Decimation means the reconstructor only gets the block-level
aggregate — that's what makes the comparison fair across methods and meaningful
as "what if we could only communicate a coarser signal." `gripper_sync` violates
this for the discrete dim specifically, which makes it incomparable to every
number already in every table in this repo, and would be an immediate, obvious
methodology objection from any reviewer.

## The fix

`gripper_sync` must reconstruct from a per-block SUMMARY of the raw gripper trace,
never the raw per-step trace itself, matching `coarsen_delta`'s role for the
continuous dims:

1. Add `coarsen_gripper_transitions(raw_col, k) -> np.ndarray` (int array, one
   entry per block): for each k-step block, find where `raw_col` differs from the
   block's starting value; record the LAST such offset within the block (an int in
   `[0, k-1]`, or `-1` if the value never changes within the block). This is
   exactly one integer per block — the same order of information `block_sum`
   provides per block for a continuous dim, not the raw trace.
2. Rewrite `gripper_sync(chunk, transition_offsets, n_hold, k)` to take that
   coarsened array (NOT `raw_chunk`) as its second argument. Reconstruct each
   block: hold the block's starting value (whatever `apply_arm`'s existing
   causal-hold already produces for step 0 of the block) up to the recorded
   offset, then switch to the block's ending value from that offset onward. If
   offset is -1, the whole block stays at the starting value (identical to plain
   causal-hold — correct, since no transition happened).
3. In `apply_arm`, compute `coarsen_gripper_transitions(chunk[:, gripper_dim], K)`
   once (this is fine — it's the same kind of one-time coarsening `coarsen_delta`
   already does on `chunk` for the continuous dims) and pass that summary, not
   `chunk` itself, into `gripper_sync`.
4. Rewrite `test_gripper_sync.py` and `eval_gripper_sync_openloop.py` against the
   new signature. The reconstruction-accuracy numbers will very likely drop from
   ~100% to something more modest — report the real numbers, whatever they are.
   A smaller or null advantage here is a legitimate, reportable finding — it is
   not a failed exercise, it's what honest measurement looks like.
5. Re-run `eval_gripper_sync_openloop.py --k 4` and update `PROPOSAL.md`'s
   open-loop table with the corrected numbers. State plainly in the doc that the
   original numbers were invalid due to oracle access and have been corrected.

## Also: attempt the closed-loop run again

The sandbox blocked `robocasa_bridge.py` from binding a localhost socket
(`PermissionError: [Errno 1] Operation not permitted`). Try once more with
`-s danger-full-access` on the codex exec invocation if that flag exists in your
environment; if it still fails, say so explicitly again — do not fabricate
closed-loop numbers. I'm separately attempting the bridge myself outside your
sandbox in parallel.

## Do NOT

- Do not keep any table in PROPOSAL.md that used the old oracle-based
  gripper_sync numbers — replace them, don't append caveats next to them.
- Do not change any other resampler (zoh/spline/tac_fold/pchip/bspline/qp).
