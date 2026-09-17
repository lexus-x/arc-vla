# Fable review, round 2: oracle bug is fixed, but the offset is the wrong endpoint

Good: `gripper_sync` no longer reads the raw per-step trace directly, and the new
test correctly asserts `synced != raw`. The oracle-access bug from round 1 is
fixed. But there's a new, separate correctness bug in how the offset is computed
and used.

## The bug

`coarsen_gripper_transitions` (resample_math.py) computes, per block:
`offsets = np.where(blocks != blocks[:, :1], np.arange(k), -1).max(axis=1)` — the
**last** index that differs from the block's *starting* value. For almost any
real single, sticking transition (value changes once and stays changed through
the end of the block — the common real case), this evaluates to `k-1` regardless
of where in the block the transition actually happened.

Then `gripper_sync` writes only `out[block_start + offset : block_stop]` — since
`offset` is nearly always `k-1`, this only ever overwrites the block's **single
last native step**. Every step between the true transition point and the block's
end is left at the old causal-hold value — silently wrong, not reconstructed.

I traced this by hand against your own updated harness-level test
(`test_harness_arm_defaults_to_spline_satfix_and_uses_coarsened_gripper_timing`):
raw block1 = `[1, -1, -1, -1]` (transition at position 1, stays -1 through the
block). True reconstruction should be `[1, -1, -1, -1]`. Your code produces
`[1, 1, 1, -1]` — only the last step (-1) is correct; positions 1 and 2 are wrong
(still held at the block's start value, 1). The test passes only because it
asserts against `[-1, -1, -1, 1, 1, 1, 1, -1]`, which was written to match this
output, not the true raw signal — the test is validating the bug, not the
behavior.

## The fix

Redefine the per-block summary to find the **first** index of the block's
**final run** (the maximal trailing span that all equal the block's *last* raw
value), when that final value differs from the block's start:

```python
def coarsen_gripper_transitions(raw_col, k):
    n_blocks = len(raw_col) // k
    blocks = raw_col[: n_blocks * k].reshape(n_blocks, k)
    end_val = blocks[:, -1:]
    start_val = blocks[:, :1]
    # first index i such that blocks[:, i:] are ALL equal to end_val
    matches_end = blocks == end_val                      # (n_blocks, k) bool
    # reverse-cumulative-all: True at i iff every position from i to k-1 matches end_val
    run_start = k - 1 - np.argmax(matches_end[:, ::-1].cumprod(axis=1)[:, ::-1][:, ::-1].cumsum(axis=1) == np.arange(1, k+1), axis=1)
    # (pick whatever correct, readable implementation you like for "first index of
    # the trailing constant run equal to blocks[:,-1]" -- the cumprod/argmax above
    # is illustrative, not mandated; a simple per-row Python loop over k is fine too,
    # k is small)
    offsets = np.where(end_val[:, 0] != start_val[:, 0], run_start, -1)
    return offsets.astype(np.int64, copy=False)
```

Don't copy that snippet verbatim if the reversed-cumprod indexing is hard to get
right — a plain per-block Python loop (`for i in range(k-1, -1, -1): if block[i] !=
end_val: break` → `run_start = i + 1`) is completely fine; `k` is at most a handful
so there's no performance concern. What matters: **the returned offset must be the
first step of the block that should show the new value**, not the last step that
differs from the old one.

Then in `gripper_sync`, reconstruct using the block's own recorded end value —
don't infer it as `-start_value` (that assumes strictly binary ±1 signals and will
silently break for anything else). Either pass the end value alongside the offset
(e.g. return `(n_blocks, 2)`: `[offset, end_value]` per block, or two parallel
arrays), or have `gripper_sync` read `chunk`'s own gripper column at the block's
last index for the value to write (that's fine — it's the coarsened per-block
*value*, not the raw per-step trace, so it doesn't reintroduce the round-1 bug).

## Verify

Re-derive the harness-level test by hand (or add an assertion against the literal
raw array, not a hardcoded expected array) so a future regression can't silently
re-introduce a "matches whatever the code does" test again. Re-run
`eval_gripper_sync_openloop.py --k 4` and update `PROPOSAL.md`'s open-loop table
with the corrected numbers.

## Do NOT

- Do not reintroduce reading `raw_chunk`'s per-step values in `gripper_sync`
  itself — only the coarsened per-block offset+value summary.
- Do not touch anything closed-loop related; I'm running that separately.
