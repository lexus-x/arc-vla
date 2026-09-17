# Task 1 — Stage 1 data acquisition and validation

Read this first: it is your requirements, with the exact values to use verbatim.

Implement the data-acquisition portion of PLAN.md Stage 1 as production-quality code in this
workspace. Add a Python CLI (preferred over a shell-only script for testability) that:

1. Downloads RoboCasa `human_im` HDF5 datasets into a new `robocasa_data_vision/` directory,
   never touching or overwriting `robocasa_data/*_ld.hdf5`.
2. Covers exactly these four task/URL pairs:
   - `TurnOffSinkFaucet`: `https://utexas.box.com/shared/static/ceewfn4ydhprupdcdppfe8wu4x61oxdg.hdf5`
   - `CoffeePressButton`: `https://utexas.box.com/shared/static/l5dnmcfd0r36vhdqgjchxo20vajt7ohl.hdf5`
   - `TurnOffMicrowave`: `https://utexas.box.com/shared/static/0drm2h7fgd5857x8xgcj1lph23srpbj1.hdf5`
   - `CloseSingleDoor`: `https://utexas.box.com/shared/static/2wnm0u1x9fp9ni02pmzqzhpjsb1kgfrr.hdf5`
3. Uses redirects and the browser user-agent `Mozilla/5.0`. Make interrupted downloads
   resumable and atomic: download to a partial path and rename only after HDF5 validation.
4. Opens every completed file with `h5py`, parses `data.attrs['env_args']`, requires
   `use_camera_obs: True`, discovers and reports camera image keys present under demo `obs`,
   and reports/confirms the demo count (`len(f['data'].keys())`). It must reject a file with no
   RGB image observations.
5. Supports verification without re-downloading and selection of one or all tasks, with clear
   CLI help and non-zero failures.
6. Add focused automated tests using tiny synthetic HDF5 files; tests must not require network.

Do not alter `harness.py`. Do not begin Stage 2 or reference/download `mg_im` data.

