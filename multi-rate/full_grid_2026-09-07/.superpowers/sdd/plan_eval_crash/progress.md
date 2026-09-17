# SDD ledger — plan: PLAN.md (2026-09-10 15:53 eval-crash revision)

Repository context: the supplied workspace has an empty, non-functional `.git`
directory. Worktree creation, commits, and commit-based review packages are therefore
unavailable; this run uses pre-change snapshots and unified diffs.

No separate specification is referenced by this revision of `PLAN.md`; `PLAN.md` is
the binding requirements source.

## Pre-flight interface scan

| Items | Producer / consumer interface | Finding |
|---|---|---|
| Bug 1 / eval verification | NumPy 2.2.6 pickle producer -> NumPy 1.23.3 `recv_msg` consumer | Compatible only after the requested `numpy._core` aliases are installed before importing and calling `recv_msg`; the baseline real-image-array round trip fails with `ModuleNotFoundError`. |
| Bug 1 / state-only bridge prohibition | Compatibility shim placement -> shared `robocasa_bridge.recv_msg` | Put the shim at the top of `robocasa_vision_bridge.py`, keeping `robocasa_bridge.py` byte-for-byte untouched. |
| Bug 2 / eval teardown | `VisionTaskState.env` wrapper -> underlying simulator close | The current file already uses `wrapper.env.close()`, matching the installed wrapper shape and the prior focused cleanup test. No further close-path edit is required. |
| Bug 1 internally | Requested aliases -> NumPy 1.23.3 module layout | Self-consistent: aliases are conditional and map only available `numpy.core` modules. |
| Bug 2 internally | Correct simulator close -> HDF5 cleanup | Self-consistent: simulator cleanup targets the underlying env and data cleanup is preserved even on simulator-close failure. |
| Verification internally | Existing checkpoint -> eval-only 15-episode result | Self-consistent: invoke `stage1_vision.py TurnOffSinkFaucet --mode eval` with the saved checkpoint and default 15-example held-out split; do not invoke the campaign's train-eval path. |

Ruling: Treat the already-present `wrapper.env.close()` implementation as fulfillment of
Bug 2 rather than rewriting equivalent code — unnecessary rewriting risks changing tested
cleanup semantics; if the current wrapper structure differs at runtime, eval teardown will
still expose that mismatch.

Baseline: policy NumPy 2.2.6 produced a `(3, 8, 9)` uint8 image pickle; bridge
NumPy 1.23.3 failed to load it with `ModuleNotFoundError: numpy._core`. Focused
suite before editing: `30 passed in 1.84s`.

Task 1 implementation review: spec compliant, quality approved, no findings.

Task 1 code complete (snapshot diff review clean). Post-change producer/consumer
round trip passed for an exact `(3, 16, 17)` uint8 image under NumPy 2.2.6 ->
1.23.3. Full suite: `50 passed in 3.66s`. Protected file/checkpoint hashes match.

Eval verification blocked by host policy before application behavior: the fresh
bridge cannot create an AF_INET socket (`PermissionError: [Errno 1] Operation not
permitted`). A socket-free environment construction attempt then reached RoboCasa
but could not write its generated MJCF into the read-only external installation
tree (`OSError: [Errno 30] Read-only file system`). No episode ran and no result
JSON was created.

Final whole-change review: no code defects; ready to merge. The missing 15-episode
result remains an Important acceptance-verification gap caused by host policy.

Ruling: Do not alter the TCP transport or RoboCasa installation behavior to evade
the managed sandbox — both would exceed this crash-fix plan and would not verify the
production eval path — cost if wrong: acceptance remains pending until the exact
eval-only command is run on the normal writable host.

Task 1: code complete (snapshot reviews clean); live acceptance blocked by host policy.
