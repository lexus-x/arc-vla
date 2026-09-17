# Task 2 — Stage 1 vision policy, bridge, training, and native evaluation

Read this first: it is your requirements, with the exact values to use verbatim.

Implement PLAN.md Stage 1 items 2–5, integrating with Task 1's validated
`robocasa_data_vision/<Task>.hdf5` files.

Requirements:

1. In `dp_min.py`, add an additive visual Diffusion Policy variant. Keep `DiffusionPolicy`,
   `train`, and all existing state-only APIs working unchanged. Use a standard small ResNet,
   specifically torchvision `resnet18`, remove its classification FC and add a fixed-size
   learned projection. Process each camera image at each observation timestep, concatenate
   the camera embeddings with non-privileged proprioceptive state, and use that as the U-Net's
   global conditioning. Do not use pretrained/network-fetched weights.
2. The vision path must accept raw image arrays/tensors robustly (uint8 images from HDF5/live
   env), convert to float, map to the conventional image range, handle HWC/CHW deliberately,
   and preserve separate state/image inputs so pixels are never min-max-normalized as flat state.
3. Proprioception must include eef pose and gripper state and must exclude the privileged
   simulator `object` observation. Use explicit keys and keep stored/live key mapping coherent.
4. Update `robocasa_bridge.py` additively (or add a vision-specific bridge) so a vision mode
   reads the new HDF5 files, harvests raw image arrays plus proprioception and actions, and
   returns matching live image/proprioception on reset/step. The current state-only mode and
   `robocasa_data/*_ld.hdf5` behavior must remain unchanged. Camera names used to create the
   environment must be derived from validated dataset image keys / metadata, not hard-coded
   to an unrelated camera. Stored and live vertical orientation must be made consistent.
5. Add a separate Stage 1 training/evaluation CLI rather than changing `harness.py`. It must
   support all four Table 2a tasks and default to a disjoint 35-train / 15-eval split for each
   50-demo file, while allowing counts to be overridden. Use the prior budget class defaults:
   15,000 optimizer steps and batch size 256, with existing n_obs=2, horizon=16,
   n_action_steps=8, EMA, and DDIM-10 behavior. Do not tune task-specific hyperparameters.
6. Save self-describing checkpoints containing EMA policy weights, policy architecture values,
   state/action normalizer states, proprioception keys, camera image keys, train/eval demo
   indices, and training budget/seed. Reject incompatible checkpoint/data metadata on load.
7. Native closed-loop evaluation must execute the policy's first 8 predicted actions directly
   at k=1 with no decimation/resampling, use at least 15 episodes by default, report progress,
   and write structured JSON. Compare success plainly against Diff.1X Base paper values:
   SinkFaucet 79%, CoffeePressButton 93%, Microwave 77%, CloseDoor 27%. Include counts,
   percentages, percentage-point gaps, task mapping, checkpoint/data identity, and explicitly
   state that demo count is the next likely cause when results remain far below the paper.
8. Provide a campaign script/command that runs the bridge plus the four tasks sequentially,
   preserving logs and stopping cleanly. It must not start Stage 2.
9. Add focused CPU-feasible tests for tensor contracts, multimodal chunk alignment,
   object-key exclusion, image orientation/preprocessing, checkpoint metadata validation, and
   paper-comparison/result formatting. Tests must use small dimensions/backbones where needed;
   they must not require RoboCasa, network, CUDA, or full training.

Global constraints: do not touch `harness.py`; do not overwrite `robocasa_data/*_ld.hdf5`;
keep every existing state-only path intact; do not implement/start Stage 2; do not claim a
scientific win without real four-task native n>=15 evaluation.
