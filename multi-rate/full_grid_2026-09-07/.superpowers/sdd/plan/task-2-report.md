# Task 2 report — Stage 1 vision policy, bridge, training, and native evaluation

## Status

Implementation is complete and CPU-feasible verification passes. The four real Task 1
datasets were validated and opened through the new Stage 1 loader. Per the task boundary,
no training, native evaluation, or Stage 2 work was run, so there is no scientific success
claim. This managed sandbox denies localhost socket creation, which prevents an end-to-end
bridge startup/rollout here; the standalone bridge CLI and its dependencies were verified,
and the transport-independent bridge/data and rollout contracts are covered by focused tests.

The workspace is not a usable Git repository (`fatal: not a git repository`), so the required
pre-task snapshot at `/tmp/stage1_task2_before_2` was used for the before/after audit.

## Files changed

- `dp_min.py` — additive `VisualDiffusionPolicy`, explicit RGB preprocessing, shared
  untrained torchvision ResNet-18 encoder with its FC removed, and learned fixed-size
  projection. The existing `DiffusionPolicy`, `train`, and state-only APIs are unchanged.
- `robocasa_vision_data.py` — validated HDF5/live multimodal contract, explicit
  non-privileged proprioception, dataset-derived camera configuration, identity metadata,
  deterministic disjoint split, and lazily materialized aligned episode windows.
- `robocasa_vision_bridge.py` — separate trusted-local RoboCasa vision service for metadata,
  raw trajectory harvesting, reset-to-demo, and native step requests.
- `stage1_vision.py` — separate Stage 1 train/eval CLI, effective-batch gradient
  accumulation, EMA checkpointing, strict metadata validation, native k=1 rollout, progress,
  and structured paper-comparison JSON.
- `run_stage1_vision_campaign.sh` — cleanly stopping, resumable, sequential four-task Stage 1
  campaign with an owned bridge and append-only per-task logs.
- `test_stage1_vision.py` — 18 focused, CPU-only tests.
- `.superpowers/sdd/plan/task-2-report.md` — this report.

Not changed:

- `harness.py` was not edited.
- `robocasa_bridge.py` is byte-identical to `/tmp/stage1_task2_before_2/robocasa_bridge.py`.
- No `robocasa_data/*_ld.hdf5` file was written; their existing mtimes and final SHA-256
  values are recorded below.

## Requirement mapping

1. **Additive visual DP:** `VisualDiffusionPolicy` subclasses the existing policy, preserving
   all state-only entry points. Its production default is `torchvision.models.resnet18` with
   `weights=None`; `fc` is replaced by `nn.Identity`, followed by a learned `nn.Linear`
   projection. Each ordered camera is encoded at every observation timestep, then camera
   embeddings and proprioception are concatenated as the U-Net global condition. The `tiny`
   encoder is explicit and exists only to make focused CPU tests small.
2. **Robust image path:** `prepare_rgb` requires an explicit HWC/CHW layout, validates RGB
   channel position and finite `[0,255]` input values, accepts NumPy arrays or tensors and
   uint8 or float ranges, converts to float `[0,1]`, and moves channels without spatial
   flipping. State and image arguments remain separate through windows, training, and sample.
3. **Non-privileged proprioception:** the only state keys are
   `robot0_eef_pos`, `robot0_eef_quat`, and `robot0_gripper_qpos` (actual width 3+4+2).
   `object` is never included. Stored and live observations use the same explicit ordered keys.
4. **Vision bridge:** the separate bridge reads only `robocasa_data_vision` by default and
   leaves the original bridge/state-only datasets untouched. `VisionDataset.camera_config`
   derives `robot0_agentview_left`, `robot0_agentview_right`, and `robot0_eye_in_hand` from the
   validated `*_image` keys and the stored image shapes. Installed RoboCasa's
   `EnvRobocasa.get_observation` applies `di[k][::-1]`, while its data-processing constructor
   uses `postprocess_visual_obs=False`; dataset generation and live bridge observations
   therefore use the same wrapper flip. No additional flip is applied.
5. **Separate Stage 1 CLI:** `stage1_vision.py` supports exactly the four Table 2a tasks.
   Defaults are 35 train / 15 held-out eval, 15,000 optimizer steps, effective batch 256,
   `n_obs=2`, `horizon=16`, `n_action_steps=8`, EMA, and DDIM-10. Counts and budgets have CLI
   overrides; no task-specific training hyperparameters exist.
6. **Self-describing checkpoints:** atomic saves contain EMA weights, every constructor
   architecture value, dataset SHA-256 and modality/shape metadata, state/action normalizers,
   proprio/camera keys, train/eval indices, step/batch/microbatch/LR/EMA/DDIM budget, and seed.
   Loads validate format, exact data identity/metadata, modality dimensions/layouts, Stage 1
   temporal settings, budget/seed, split validity, and normalizer ranges before strict weight
   loading.
7. **Native evaluation and comparison:** `native_rollout` samples with DDIM-10 and passes the
   first eight chronological actions directly to eight consecutive bridge `step` calls—no
   resampling or decimation. The default checkpoint split has 15 episodes. Each episode prints
   progress and atomically updates JSON. Results contain task mapping, counts, percentages,
   signed percentage-point gap, all four exact paper references (79/93/77/27), checkpoint
   path/hash/architecture, dataset path/hash/metadata, splits, and budget. The interpretation
   explicitly names demo count as the next likely cause if results remain far below the paper.
8. **Sequential campaign:** `run_stage1_vision_campaign.sh` owns one port/bridge, runs
   TurnOffSinkFaucet, CoffeePressButton, TurnOffMicrowave, then CloseSingleDoor sequentially,
  appends bridge/task logs, skips only matching complete results, resumes evaluation when a
  checkpoint exists, gives Numba an explicit writable cache directory, and traps normal
  exit/INT/TERM to stop its current policy and bridge. It invokes no Stage 2 code.
9. **Focused tests:** `test_stage1_vision.py` covers visual and retained state-only tensor
   contracts, actual ResNet-18 construction/forward, HWC/CHW/range/orientation preprocessing,
   multimodal edge padding/alignment and rejection, object exclusion, live/stored orientation,
   camera derivation, the 35/15 split, checkpoint incompatibilities and round-trip contents,
   direct first-eight native actions, and all four paper-result formats. Tests use synthetic
   HDF5 and tiny U-Net/backbone dimensions where appropriate and require no RoboCasa, CUDA,
   network, or training run.

## Self-review

- Checked every requirement in `task-2-brief.md` against the final source and tests.
- Confirmed `resnet18(weights=None)` prevents pretrained/network-fetched weights.
- Confirmed image values are not passed through `MinMax`; only state and actions are.
- Confirmed window indices never cross episode boundaries and leading state/image/action
  timesteps match, including start/end edge padding.
- Confirmed all actual datasets have 54 demos, state width 9, action width 12, three HWC uint8
  128x128 cameras, and the same camera configuration returned to the bridge.
- Confirmed native actions are not sent through any decimator/resampler.
- Confirmed checkpoint validation happens before policy construction/strict state loading.
- Confirmed interrupted evaluation retains the last atomic JSON progress record; a subsequent
  campaign invocation uses the existing checkpoint rather than retraining.
- Confirmed the campaign refuses to take over an already-listening port and only terminates a
  bridge PID that it started.
- Confirmed the RoboCasa environment imports successfully when given the same explicit writable
  Numba cache location that the campaign supplies.
- Confirmed no scientific success statement is emitted solely from code/test completion.

## Verification commands and outputs

### Focused tests, syntax, and full suite

```text
$ python -m py_compile dp_min.py robocasa_vision_data.py robocasa_vision_bridge.py stage1_vision.py test_stage1_vision.py
(no output; exit 0)

$ bash -n run_stage1_vision_campaign.sh
(no output; exit 0)

$ python -m pytest -q test_stage1_vision.py
..................                                                       [100%]
18 passed in 2.08s

$ python -m pytest -q
......................................                                   [100%]
38 passed in 3.18s
```

The intended policy environment also passes the focused suite:

```text
$ /home/user/anaconda3/envs/vla_smolvla_libero/bin/python -m pytest -q test_stage1_vision.py
..................                                                       [100%]
18 passed in 1.76s

$ /home/user/anaconda3/envs/vla_smolvla_libero/bin/python -m py_compile dp_min.py robocasa_vision_data.py robocasa_vision_bridge.py stage1_vision.py test_stage1_vision.py
(no output; exit 0)
```

### Real dataset validation

```text
$ python download_robocasa_vision_data.py --verify
[valid] TurnOffSinkFaucet: /home/user/Desktop/multi-rate/full_grid_2026-09-07/robocasa_data_vision/TurnOffSinkFaucet_human_im.hdf5
        demos: 54
        camera image keys: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
        RGB image keys: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
[valid] CoffeePressButton: /home/user/Desktop/multi-rate/full_grid_2026-09-07/robocasa_data_vision/CoffeePressButton_human_im.hdf5
        demos: 54
        camera image keys: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
        RGB image keys: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
[valid] TurnOffMicrowave: /home/user/Desktop/multi-rate/full_grid_2026-09-07/robocasa_data_vision/TurnOffMicrowave_human_im.hdf5
        demos: 54
        camera image keys: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
        RGB image keys: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
[valid] CloseSingleDoor: /home/user/Desktop/multi-rate/full_grid_2026-09-07/robocasa_data_vision/CloseSingleDoor_human_im.hdf5
        demos: 54
        camera image keys: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
        RGB image keys: robot0_agentview_left_image, robot0_agentview_right_image, robot0_eye_in_hand_image
```

Direct loader output on all four files:

```text
TurnOffSinkFaucet 54 9 12 {'camera_names': ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'], 'camera_height': 128, 'camera_width': 128} robocasa_data_processing_wrapper_no_extra_flip
CoffeePressButton 54 9 12 {'camera_names': ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'], 'camera_height': 128, 'camera_width': 128} robocasa_data_processing_wrapper_no_extra_flip
TurnOffMicrowave 54 9 12 {'camera_names': ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'], 'camera_height': 128, 'camera_width': 128} robocasa_data_processing_wrapper_no_extra_flip
CloseSingleDoor 54 9 12 {'camera_names': ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'], 'camera_height': 128, 'camera_width': 128} robocasa_data_processing_wrapper_no_extra_flip
```

### Existing state-only smoke and preservation

```text
$ python dp_min.py
dp_min smoke OK on cpu; params=64.2M

$ cmp -s robocasa_bridge.py /tmp/stage1_task2_before_2/robocasa_bridge.py && echo 'robocasa_bridge.py unchanged from snapshot'
robocasa_bridge.py unchanged from snapshot

$ sha256sum robocasa_data/*_ld.hdf5 harness.py | sort
00a4e6e4391fa6ddf04018a268b6064359b9124908dd6ffe943ce43e5fff30b4  robocasa_data/OpenDrawer_ld.hdf5
1f37fc5abc93a8a8f57b4d5a7c844e061741f36b26265d89ade1e315c5287093  robocasa_data/TurnOffSinkFaucet_ld.hdf5
36c74d0b9462be43cb3d1ba3cf513d8f0540d67abc53be12e3a1061dc00ae5fd  robocasa_data/PnPCounterToStove_ld.hdf5
af073c6955b02fa895633176b2c1d4a2907f66ef7b461449c1bd211fd086c9a7  robocasa_data/CloseSingleDoor_ld.hdf5
afc1e7eb99dbda08f9a21aa63be8a8c761d875b365e7d572c623f187ca82d13b  robocasa_data/CoffeePressButton_ld.hdf5
ed0aad473c674c90037ed2f03e9bba8a063fdc25adc76bbae226b30c5ac013bc  robocasa_data/TurnOffMicrowave_ld.hdf5
f6888a70e205d51a44662fa5b055ccac751a41649e4bc81106a66579d0921d16  harness.py
```

## Final review fix wave — evaluation horizon and bridge cleanup

### Important findings addressed

> 1. Campaign reuse ignores evaluation step limit — stage1_vision.py:124 and :172 omit max_steps. A completed 15-episode run with --max-steps 1 is accepted by the default campaign. Include the effective task-specific max_steps in the canonical request, require result max_steps equality, have CLI/campaign pass the same resolved/requested limit, and add a short-horizon-to-default reuse regression. Preserve current task defaults (CoffeePressButton 300, other tasks 500) and allow the existing CLI override coherently; expose a campaign MAX_STEPS override only if it can remain coherent across tasks, otherwise keep task-resolved defaults.

> 2. Bridge cleanup calls unsupported wrapper method — robocasa_vision_bridge.py:52 calls self.env.close(), but installed EnvRobocasa has no close/forwarding; supported simulator is self.env.env.close(). Close the underlying simulator safely, release the wrapper reference, always close HDF5, and add a wrapper-shaped cleanup regression that confirms underlying close is invoked without masking a failure. Ensure sequential campaign shutdown is clean.

### Fixes

- Added `default_max_steps(task)` and included the resolved positive integer in the canonical
  experiment request. It preserves 300 steps for `CoffeePressButton` and 500 for the other
  three tasks; an explicit `--max-steps` overrides either default.
- Both artifact preflight and live evaluation now consume the same canonical request value.
  `validate_result_request` requires exact `result.max_steps` equality, so a completed
  short-horizon result cannot be reused by the default campaign.
- Added optional campaign `MAX_STEPS`. When unset, each task resolves its existing task-specific
  default inside the CLI. When set, the same global override is passed to both preflight and
  execution for every task, keeping artifact validation and rollout coherent.
- Changed `VisionTaskState.close` to detach the wrapper reference first and call the supported
  `wrapper.env.close()` on the underlying robosuite simulator. HDF5 close is always attempted.
  If simulator teardown fails, that original failure remains primary even if HDF5 teardown also
  fails; the latter is chained as its cause rather than masking it.
- Added three focused tests: a completed `max_steps=1` result is rejected by a default
  `max_steps=500` request; CoffeePressButton/other-task defaults and an explicit override are
  exact; and a wrapper-shaped cleanup double proves underlying simulator close and HDF5 close
  are both invoked, the wrapper reference is released, and the simulator failure is preserved.

### Final-wave self-review

- Traced `args.max_steps` through canonical request creation, artifact validation, rollout, and
  result serialization. There is now one resolved effective value per task and no fallback
  recomputation at evaluation time.
- Traced campaign settings through both invocations of `stage1_vision.py`; optional
  `MAX_STEPS_ARGS` is identical for preflight and execution, while the empty case lets both
  independently resolve the same task default.
- Confirmed complete-result reuse requires exact horizon equality in addition to the prior
  task/data/checkpoint/split/budget/seed/count checks.
- Confirmed installed `EnvRobocasa` has no `close` method and the installed underlying
  `robosuite.environments.base.MujocoEnv` exposes `close`.
- Confirmed cleanup releases `VisionTaskState.env` before teardown, calls the underlying
  simulator, attempts HDF5 closure on success or failure, and preserves the simulator error if
  both closures fail. This path is used by per-task client close and final bridge shutdown, so
  sequential campaign teardown no longer depends on an unsupported wrapper method.
- Confirmed `harness.py`, the state-only bridge, and state-only datasets remain unchanged.
- Did not touch the user-owned `stage1_vision_logs` / `stage1_vision_results` directories or any
  running process during this wave. No training, evaluation, bridge, or Stage 2 action was run.

### Final-wave verification commands and exact outputs

```text
$ python -m py_compile stage1_vision.py robocasa_vision_bridge.py test_stage1_vision.py
(no output; exit 0)

$ bash -n run_stage1_vision_campaign.sh
(no output; exit 0)

$ python -m pytest -q test_stage1_vision.py
..............................                                           [100%]
30 passed in 1.86s

$ python -m pytest -q
..................................................                       [100%]
50 passed in 3.44s

$ /home/user/anaconda3/envs/vla_smolvla_libero/bin/python -m pytest -q test_stage1_vision.py
..............................                                           [100%]
30 passed in 1.85s

$ ./run_stage1_vision_campaign.sh --help
Usage: [POLICY_PY=...] [BRIDGE_PY=...] [PORT=8766] [MAX_STEPS=...] [FORCE=0|1] ./run_stage1_vision_campaign.sh
Runs only the four Stage 1 visual-policy tasks, sequentially. Other defaults may also be overridden by environment variable.
FORCE=1 re-evaluates a compatible completed checkpoint; incompatible artifacts require a distinct OUTPUT_DIR.
```

Installed cleanup API check:

```text
$ rg -n '^    def close\(' /home/user/Isaac-GR00T/external_dependencies/robocasa/robocasa/utils/robomimic/robomimic_env_wrapper.py /home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/lib/python3.10/site-packages/robosuite/environments/base.py
/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/lib/python3.10/site-packages/robosuite/environments/base.py:784:    def close(self):
```

Preservation check:

```text
$ cmp -s robocasa_bridge.py /tmp/stage1_task2_before_2/robocasa_bridge.py && echo 'robocasa_bridge.py unchanged'
robocasa_bridge.py unchanged

$ sha256sum harness.py robocasa_data/*_ld.hdf5 | sort
00a4e6e4391fa6ddf04018a268b6064359b9124908dd6ffe943ce43e5fff30b4  robocasa_data/OpenDrawer_ld.hdf5
1f37fc5abc93a8a8f57b4d5a7c844e061741f36b26265d89ade1e315c5287093  robocasa_data/TurnOffSinkFaucet_ld.hdf5
36c74d0b9462be43cb3d1ba3cf513d8f0540d67abc53be12e3a1061dc00ae5fd  robocasa_data/PnPCounterToStove_ld.hdf5
af073c6955b02fa895633176b2c1d4a2907f66ef7b461449c1bd211fd086c9a7  robocasa_data/CloseSingleDoor_ld.hdf5
afc1e7eb99dbda08f9a21aa63be8a8c761d875b365e7d572c623f187ca82d13b  robocasa_data/CoffeePressButton_ld.hdf5
ed0aad473c674c90037ed2f03e9bba8a063fdc25adc76bbae226b30c5ac013bc  robocasa_data/TurnOffMicrowave_ld.hdf5
f6888a70e205d51a44662fa5b055ccac751a41649e4bc81106a66579d0921d16  harness.py
```

### CLI and bridge-environment checks

```text
$ ./run_stage1_vision_campaign.sh --help
Usage: [POLICY_PY=...] [BRIDGE_PY=...] [PORT=8766] [FORCE=0|1] ./run_stage1_vision_campaign.sh
Runs only the four Stage 1 visual-policy tasks, sequentially. Other defaults may also be overridden by environment variable.

$ /home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python robocasa_vision_bridge.py --help
usage: robocasa_vision_bridge.py [-h] [--port PORT] [--data-dir DATA_DIR]
...

$ /home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python - <<'PY'
import h5py, numpy
print('bridge deps', h5py.__version__, numpy.__version__)
PY
bridge deps 3.16.0 1.23.3

$ NUMBA_CACHE_DIR=/tmp/task2_numba_cache /home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python - <<'PY'
import robocasa
print('robocasa import OK', robocasa.__file__)
PY
[robosuite WARNING] No private macro file found! (macros.py:57)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:58)
[robosuite WARNING] To setup, run: python /home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py (macros.py:59)
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly (e.g. pip install mink==0.0.5), otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[robosuite WARNING] No private macro file found! (macros.py:28)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:29)
[robosuite WARNING] To setup, run: python /home/user/Isaac-GR00T/external_dependencies/robocasa/robocasa/scripts/setup_macros.py (macros.py:30)
WARNING: mimicgen environments not imported since mimicgen is not installed!
robocasa import OK /home/user/Isaac-GR00T/external_dependencies/robocasa/robocasa/__init__.py
```

### Sandbox-limited check

An early campaign help smoke, before explicit `--help` handling was added, attempted only to
bind the bridge socket and failed immediately:

```text
PermissionError: [Errno 1] Operation not permitted
```

No RoboCasa environment was created and no training/evaluation process ran. The generated
empty result directory and startup log were removed. Final `--help` now exits before any
side effect. A real bridge/native-rollout run remains intentionally pending outside the
managed sandbox.

## Fix round 1/5 — campaign artifact compatibility

### Important finding addressed

> "Campaign reuse does not verify that existing artifacts match the requested experiment (run_stage1_vision_campaign.sh:112, run_stage1_vision_campaign.sh:133). Completion checks only task, stage, checkpoint path, and file existence. A completed one-episode smoke campaign is therefore skipped by a later default 15-episode campaign; replacing checkpoint contents at the same path also leaves stale results accepted. If evaluation runs, existing checkpoints silently determine the split and training budget despite the campaign passing new settings (stage1_vision.py:275). Validate checkpoint/data hashes, requested budget and split, evaluation seed, and episode count before reuse. Reject incompatible artifacts clearly or use a distinct output location. Add focused coverage for smoke-to-production reuse and changed checkpoint contents."

### Fix

- Added a canonical `experiment_request` in `stage1_vision.py`. It resolves the deterministic
  requested train/eval indices from the requested counts and records task, current dataset
  SHA-256, optimizer steps, batch size, microbatch, training seed, and evaluation seed.
- Added `validate_checkpoint_request`. In addition to the existing exact checkpoint/data
  metadata validation, it requires the checkpoint's train and eval indices to equal the
  requested split and requires exact requested steps, batch size, microbatch, and training
  seed. Direct `--mode eval` now calls this validation before constructing the policy or
  connecting to the bridge, so CLI options cannot be silently ignored in favor of an existing
  checkpoint.
- Added `validate_result_request`. Result reuse now requires the requested task and task
  mapping, resolved checkpoint path, SHA-256 of the checkpoint's current contents, resolved
  dataset path and complete current metadata (including dataset SHA-256), requested split,
  checkpoint training budget, requested evaluation seed, internally consistent episode
  records/count, and at least the requested number of episodes for a completed result.
- Added `campaign_artifact_status`, which has three explicit outcomes: `train-eval` only when
  neither artifact exists, `eval` only for a compatible checkpoint with no complete matching
  result, and `skip` only for a fully compatible completed result. A result without its
  checkpoint, malformed/stale result, or incompatible checkpoint raises a clear error rather
  than allowing overwrite or retraining.
- Updated `run_stage1_vision_campaign.sh` to invoke `stage1_vision.py --check-artifacts` for all
  four tasks before starting the bridge. It passes all requested counts, budgets, and seeds to
  both preflight and execution. Existing incompatible artifacts stop the campaign and direct
  the operator to a distinct `OUTPUT_DIR`. `FORCE=1` only re-evaluates an otherwise compatible
  completed checkpoint. Per-task logs remain append-only, and checkpoint-only interruptions
  remain resumable as evaluation.
- Extended `test_stage1_vision.py` from 18 to 27 focused tests. New coverage exercises fresh,
  compatible-checkpoint, and compatible-completed-result decisions; rejects a completed
  one-episode smoke artifact for a larger production request; rejects same-path checkpoint
  content replacement by hash; rejects changed train count, steps, batch size, microbatch,
  training seed, evaluation seed, and insufficient completed episode count; and confirms
  direct checkpoint loading rejects a requested budget mismatch.

No separate shell-focused test file was needed: the reuse decision is centralized in Python
and exhaustively unit-tested, while the shell is syntax-checked and delegates all compatibility
decisions to `--check-artifacts` before bridge startup.

### Fix-round self-review

- **Task identity:** request creation rejects a dataset task mismatch; checkpoint metadata and
  result `task` / `task_mapping.dataset_task` must match the request.
- **Data identity/hash:** checkpoint metadata must exactly match current validated metadata,
  including `data_sha256`; result metadata and resolved data path must match the current data.
- **Checkpoint identity/hash:** result checkpoint path must resolve to the requested path and
  its stored SHA-256 must equal the hash of the checkpoint's current contents.
- **Train/eval counts and split:** requested counts are converted through the same
  deterministic `split_demos` function used for training; checkpoint and result indices must
  match exactly. Thus a 1-eval smoke checkpoint cannot satisfy a 15-eval request.
- **Training budget/seed:** checkpoint reuse requires exact requested steps, batch size,
  microbatch, and training seed. Result budget must equal that validated checkpoint budget.
- **Evaluation seed/count:** an existing result must carry the requested evaluation seed. A
  complete result must contain a consistent episode list/count with count at least the
  requested eval count. An incomplete compatible result selects evaluation rather than skip.
- **No silent overwrite:** any existing incompatible checkpoint causes preflight failure; the
  campaign never falls back to `train-eval` at that path. A stale result likewise fails rather
  than being silently accepted or replaced. The help text explains use of a distinct output
  directory.
- **Ordering and clean stop:** all artifact checks finish before the bridge starts. Existing
  append-only logging and INT/TERM cleanup remain unchanged for runnable tasks.
- Reconfirmed `harness.py`, `robocasa_bridge.py`, and `robocasa_data/*_ld.hdf5` were not changed.
- This implementer did not start training, native evaluation, a socket bridge, or Stage 2
  during this fix round. Concurrent shared-workspace logs appeared for a port-8770 bridge and
  TurnOffSinkFaucet training through step 900; no corresponding process or checkpoint/result
  was present at final inspection, and those logs were left untouched.

### Fix-round verification commands and exact outputs

```text
$ python -m py_compile stage1_vision.py test_stage1_vision.py
(no output; exit 0)

$ bash -n run_stage1_vision_campaign.sh
(no output; exit 0)

$ python -m pytest -q test_stage1_vision.py
...........................                                              [100%]
27 passed in 1.87s

$ python -m pytest -q
...............................................                          [100%]
47 passed in 3.56s

$ /home/user/anaconda3/envs/vla_smolvla_libero/bin/python -m pytest -q test_stage1_vision.py
...........................                                              [100%]
27 passed in 1.87s

$ ./run_stage1_vision_campaign.sh --help
Usage: [POLICY_PY=...] [BRIDGE_PY=...] [PORT=8766] [FORCE=0|1] ./run_stage1_vision_campaign.sh
Runs only the four Stage 1 visual-policy tasks, sequentially. Other defaults may also be overridden by environment variable.
FORCE=1 re-evaluates a compatible completed checkpoint; incompatible artifacts require a distinct OUTPUT_DIR.
```

Preservation check:

```text
$ cmp -s robocasa_bridge.py /tmp/stage1_task2_before_2/robocasa_bridge.py && echo 'robocasa_bridge.py unchanged'
robocasa_bridge.py unchanged

$ sha256sum harness.py robocasa_data/*_ld.hdf5 | sort
00a4e6e4391fa6ddf04018a268b6064359b9124908dd6ffe943ce43e5fff30b4  robocasa_data/OpenDrawer_ld.hdf5
1f37fc5abc93a8a8f57b4d5a7c844e061741f36b26265d89ade1e315c5287093  robocasa_data/TurnOffSinkFaucet_ld.hdf5
36c74d0b9462be43cb3d1ba3cf513d8f0540d67abc53be12e3a1061dc00ae5fd  robocasa_data/PnPCounterToStove_ld.hdf5
af073c6955b02fa895633176b2c1d4a2907f66ef7b461449c1bd211fd086c9a7  robocasa_data/CloseSingleDoor_ld.hdf5
afc1e7eb99dbda08f9a21aa63be8a8c761d875b365e7d572c623f187ca82d13b  robocasa_data/CoffeePressButton_ld.hdf5
ed0aad473c674c90037ed2f03e9bba8a063fdc25adc76bbae226b30c5ac013bc  robocasa_data/TurnOffMicrowave_ld.hdf5
f6888a70e205d51a44662fa5b055ccac751a41649e4bc81106a66579d0921d16  harness.py
```

## Final review completion record

The detailed final-wave finding mapping, implementation notes, and self-review are recorded in
the preceding **Final review fix wave — evaluation horizon and bridge cleanup** section. This
completion record is appended last to preserve review chronology. Final verification after all
source and test edits remained:

```text
$ python -m py_compile stage1_vision.py robocasa_vision_bridge.py test_stage1_vision.py
(no output; exit 0)

$ bash -n run_stage1_vision_campaign.sh
(no output; exit 0)

$ python -m pytest -q test_stage1_vision.py
..............................                                           [100%]
30 passed in 1.86s

$ python -m pytest -q
..................................................                       [100%]
50 passed in 3.44s

$ /home/user/anaconda3/envs/vla_smolvla_libero/bin/python -m pytest -q test_stage1_vision.py
..............................                                           [100%]
30 passed in 1.85s
```

Final self-review conclusion: the effective step limit is now part of experiment identity and
cannot be bypassed during result reuse; bridge cleanup uses the installed wrapper's supported
underlying simulator API, always attempts HDF5 cleanup, and preserves the primary teardown
failure. No unrelated project or user-owned runtime state was changed.
