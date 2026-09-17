# Task 1 report — Stage 1 data acquisition and validation

## Implemented

- Added `download_robocasa_vision_data.py`, a Python CLI for the four specified
  RoboCasa `human_im` task URLs. It writes only to the new
  `robocasa_data_vision/` destination by default and refuses an output path in
  the existing `robocasa_data/` state-only directory.
- The downloader uses `urllib`'s redirect handling and the exact browser user
  agent `Mozilla/5.0`. It downloads to `<task>_human_im.hdf5.partial`, resumes
  only when the server returns a matching HTTP Range response, safely restarts
  if the server ignores Range, validates HDF5 before `rename`, and preserves
  partial files after failures. A valid pre-existing partial is finalized
  without another request.
- Validation opens the HDF5 with `h5py`, JSON-parses `data.attrs['env_args']`,
  requires `use_camera_obs: true` (including RoboCasa's `env_kwargs` layout),
  counts `len(data.keys())`, discovers image observation keys below every demo
  `obs` group, and rejects missing/non-RGB camera observations.
- CLI supports repeated `--task` selection, all tasks by default, `--verify`
  with no network activity, `--output-dir`, useful help, and non-zero error
  exits.
- Added synthetic-fixture tests in `test_download_robocasa_vision_data.py`.
  They cover successful metadata reporting, non-vision rejection, matching
  Range resume, ignored-Range restart, validation-before-finalization, valid
  partial finalization without network, verification-only selection/failure,
  and state-only output protection.

## Files changed

- `download_robocasa_vision_data.py` (new)
- `test_download_robocasa_vision_data.py` (new)
- `.superpowers/sdd/plan/task-1-report.md` (new)

`harness.py` and all existing `robocasa_data/*_ld.hdf5` files were not altered.

## Tests run

```text
$ python -m pytest -q test_download_robocasa_vision_data.py test_gripper_sync.py test_resample_invariants.py
...................                                                      [100%]
19 passed in 1.88s

$ python -m py_compile download_robocasa_vision_data.py test_download_robocasa_vision_data.py
(exit 0; no output)

$ python download_robocasa_vision_data.py --help
usage: download_robocasa_vision_data.py [-h]
                                        [--task {TurnOffSinkFaucet,CoffeePressButton,TurnOffMicrowave,CloseSingleDoor}]
                                        [--verify] [--output-dir OUTPUT_DIR]

Download or verify RoboCasa Stage 1 human_im vision datasets.

options:
  -h, --help            show this help message and exit
  --task {TurnOffSinkFaucet,CoffeePressButton,TurnOffMicrowave,CloseSingleDoor}
                        Task to process; may be repeated. Defaults to all four
                        tasks.
  --verify              Validate completed local files only; never download.
  --output-dir OUTPUT_DIR
                        Destination directory (default:
                        /home/user/Desktop/multi-
                        rate/full_grid_2026-09-07/robocasa_data_vision).

$ python download_robocasa_vision_data.py --verify --task CoffeePressButton --output-dir /tmp/robocasa_vision_missing_check
[error] CoffeePressButton: missing completed dataset: /tmp/robocasa_vision_missing_check/CoffeePressButton_human_im.hdf5
expected verify-missing exit: 1
```

## Self-review

Reviewed the implementation against each task-brief requirement. The URL map
contains exactly the four provided pairs; the default output directory is
separate from state-only data; no network tests are required; completed data
is HDF5-validated before its atomic rename; malformed/missing camera data and
verification failures propagate as non-zero exits.

## Concerns

No real datasets were downloaded or remote endpoints exercised, as required.
The first real invocation should use `--task` (or omit it for all tasks), then
run `--verify` to record the actual camera-key and demo-count output.

## Fix round 1

Changed `run_task` so `verify_only` is checked immediately after the completed
file check, before any `.partial` validation or finalization. Consequently,
`--verify` now requires `<task>_human_im.hdf5`; it neither reads nor renames a
valid partial when that completed file is absent.

Added the focused regression test
`test_verify_mode_does_not_finalize_or_inspect_a_valid_partial` in
`test_download_robocasa_vision_data.py`. It creates a valid synthetic partial,
runs `--verify`, asserts exit status 1, and asserts that the partial bytes and
path are unchanged while no completed file appears.

```text
$ python -m pytest -q test_download_robocasa_vision_data.py
...........                                                              [100%]
11 passed in 0.27s
```
