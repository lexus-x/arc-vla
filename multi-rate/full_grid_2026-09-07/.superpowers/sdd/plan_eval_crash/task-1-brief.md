# Task 1 — repair Stage 1 vision eval crashes and verify

Implement the current `PLAN.md` exactly.

1. Add this NumPy pickle compatibility shim at the top of
   `robocasa_vision_bridge.py`, before any unpickling can happen:

   ```python
   import numpy as np
   import sys
   if not hasattr(np, "_core"):
       sys.modules["numpy._core"] = np.core
       for name in ("multiarray", "numeric", "_multiarray_umath", "umath"):
           mod = getattr(np.core, name, None)
           if mod is not None:
               sys.modules[f"numpy._core.{name}"] = mod
   ```

   Keep the shim local to the vision bridge so shared state-only bridge behavior
   remains untouched.

2. Bug 2 is already implemented in the current working file: `VisionTaskState.close`
   captures the wrapper and invokes `wrapper.env.close()` while preserving HDF5 cleanup.
   Verify it and do not replace it with a bare exception suppression or unrelated rewrite.

3. Prove the shim with a real uint8 image array pickled under
   `/home/user/anaconda3/envs/vla_smolvla_libero/bin/python` (NumPy 2.2.6) and loaded
   under `/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python`
   (NumPy 1.23.3). Check shape, dtype, and values after loading; do not merely import.

4. Run focused tests suitable for the edited bridge. Do not start the full evaluation;
   the controller will run the required eval-only 15-episode rollout after review so no
   long-lived bridge or policy process escapes task ownership.

Global constraints: do not upgrade either NumPy; do not retrain
`TurnOffSinkFaucet`; do not modify `stage1_vision.py`, `robocasa_bridge.py`, its
state-only usage, the saved checkpoint, or any running training process. Work in
`/home/user/Desktop/multi-rate/full_grid_2026-09-07`. There is no usable Git
repository, so do not commit. Write the full implementation, test, and self-review
report to `.superpowers/sdd/plan_eval_crash/task-1-report.md` and return only the
short status contract.
