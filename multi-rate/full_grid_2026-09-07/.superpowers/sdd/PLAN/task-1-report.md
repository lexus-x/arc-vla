# Task 1 report — provenance and safe table assembly

## Implementation

- `harness.py:305-306` now writes `head`, `seed`, and `fold` beside the existing
  task/policy/k provenance in every newly produced result JSON.
- Added `backfill_provenance.py`, a one-off current-directory utility. It:
  - finds sorted `result_*.json` files;
  - refuses to overwrite an existing `_pre_backfill_backup` directory;
  - copies every input byte-for-byte to `_pre_backfill_backup` before parsing or
    changing any result JSON;
  - infers `head` from `_bspline` / `_blocksum` / neither, `seed` from `_s<N>`,
    and `fold` from `_f<N>` using the harness filename encoding;
  - fills only missing provenance fields, preserving fields that are already
    present; and
  - reports the number of files changed plus final head, seed, and fold counts.
- `assemble_paper_table.py` now sorts its glob, ignores files with `n < 10` and
  all non-step heads (defaulting missing `head` to `step` for legacy files), while
  retaining the previous largest-`n` arm selection. It retains a source filename
  for each selected arm and stops with status 1 after printing every conflict
  tuple when equal-length candidate lists differ by more than 10 percentage
  points in success rate.

## Focused validation

`python3 -m py_compile harness.py assemble_paper_table.py backfill_provenance.py`
passed.

The backfill utility was run only in an isolated temporary directory, never
against repository results. The fixture covered:

- `_bspline_s12_f3` inference (`bspline`, `12`, `3`);
- `_blocksum_s2` inference (`blocksum`, `2`, `-1`), plus preservation of
  already-present `head=bspline` and `seed=99` in a separate `_s4` file (only
  its absent `fold` became `-1`);
- a file with all three fields already present (unchanged); and
- byte-for-byte equality between every original fixture and its backup copy.

The successful fixture output was:

```text
backfilled files: 2
head counts: {'bspline': 1, 'step': 2}
seed counts: {0: 1, 12: 1, 99: 1}
fold counts: {-1: 2, 3: 1}
backfill fixture assertions passed
```

A second isolated fixture explicitly covered all three filename head variants
and existing-field preservation; its assertions passed with `blocksum`,
`bspline`, and default-step paths verified.

Two isolated table fixtures also passed:

- a higher-`n` step-head file supplanted its lower-`n` counterpart, while a
  non-step head and an `n=2` smoke file were excluded; the emitted PickCube 1X
  Base and 2X ZOH cells were both `100%`;
- two equal-`n` step-head files at 0% and 100% caused exit status 1 and printed:

```text
(('dp', 2, 'PickCube-v1'), 'native', 'result_a.json', 0.0, 'result_b.json', 100.0)
```

## Self-review

Confirmed the production result files were not backfilled during this task:
there is no repository `_pre_backfill_backup` directory. No rollout was run and
none of the prohibited model/head/resampling files were edited; in particular,
`heads.decode()` is untouched.

The deliberate one-off behavior is that an existing backup directory causes an
early refusal rather than replacement, so the only original-copy rollback set
cannot be silently overwritten on a later invocation. The requested actual
backfill and paper-table audit remain for Task 2.
