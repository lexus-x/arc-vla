# Task 1 report — Stage 1 vision evaluation crash repair

## Implementation

Updated `robocasa_vision_bridge.py` with the requested local NumPy pickle compatibility shim. It runs immediately after importing NumPy and before bridge/data modules can perform unpickling. On NumPy versions without `numpy._core`, it aliases `numpy._core` and the requested core submodules to the NumPy 1.x module locations.

The existing `VisionTaskState.close` implementation was reviewed and left unchanged. It captures and clears the wrapper, calls `wrapper.env.close()`, and always performs HDF5 cleanup while preserving simulator-close failures and chaining cleanup failures.

No other source, checkpoint, training process, or shared state-only bridge behavior was modified. No commit was created because the workspace does not contain a usable Git repository.

## Verification

- NumPy producer environment: `/home/user/anaconda3/envs/vla_smolvla_libero/bin/python`, NumPy `2.2.6`.
- NumPy consumer environment: `/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python`, NumPy `1.23.3`.
- A real `uint8` array was pickled by the producer and loaded by the consumer after importing the vision bridge. The consumer assertions passed for shape `(2, 3, 4)`, dtype `uint8`, and every array value (`sum=276`).
- `python -m py_compile robocasa_vision_bridge.py` passed.
- `pytest -q test_stage1_vision.py` passed: `30 passed`.
- `pytest -q test_stage1_vision.py -k cleanup` passed: `1 passed, 29 deselected`, covering simulator and HDF5 close behavior.

The required long 15-episode evaluation was not started; its lifecycle remains with the controller.

## Self-review

The source diff contains only the requested compatibility shim. The close path matches the required implementation and was not replaced or broadened. The shim is scoped to the vision bridge and guarded so NumPy 2.x is left untouched. No concerns identified.
