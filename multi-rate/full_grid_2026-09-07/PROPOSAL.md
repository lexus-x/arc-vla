# Proposal: block-local gripper-transition synchronization

## Method

`gripper_sync` reconstructs the continuous action dimensions with the existing
`spline_satfix` resampler. The raw gripper trace is separately coarsened to one offset/value
pair per complete block: the first offset of the maximal trailing run equal to the block's
ending command, or offset `-1` when the block ends at its starting command. The reconstructor
receives only those per-block summaries, holds the block-starting gripper command up to the
recorded offset, and writes the recorded ending command from that offset onward. The other
trailing hold dimensions keep their existing causal-hold behavior. This puts the discrete
dimension under the same block-summary information constraint as the continuous dimensions.

## Diagnostic evidence

The existing RoboCasa `k=4` files have zero pre-clip saturation for every recorded arm.
They also show zero episode-level discordance among `zoh`, `spline_satfix`, and `qp`: all
three produce the same 15 paired outcomes on both `RC-OpenDrawer` and
`RC-PnPCounterToStove`. On OpenDrawer, native succeeds on 6/15 episodes (40.0%) while each
continuous-resampler arm succeeds on 4/15 (26.7%), leaving a 13.3 percentage-point
decimation penalty that continuous reconstruction did not recover. PnPCounterToStove is
1/15 (6.7%) for every arm.

The task-specific transition check used the underlying `robocasa_data/*_ld.hdf5` action
arrays because the derived `.npz` caches for these two legacy cells are absent. On the
first 39 demonstrations—the training count used by the existing run—`k=4` causal holding
changes the gripper trace in 234/8,151 OpenDrawer 8-step windows (2.87%) and 495/11,955
PnPCounterToStove windows (4.14%). Transitions occur at all three possible mid-block
offsets on both tasks. The proposed mechanism is therefore exercised by the benchmark.

## Open-loop gripper reconstruction, k=4

The table uses every unique trajectory represented in the cached fold files for the four
cached RoboCasa tasks (54 demonstrations per task). Windows are the harness's edge-padded
8-step execution chunks. Accuracy is the fraction of reconstructed gripper steps exactly
matching the raw demonstrated command. The continuous resampler cannot affect this metric,
so the three baseline columns share the current causal-hold reconstruction.

These numbers replace the original oracle-based results. The original `gripper_sync` was
given the same full undecimated gripper trace used as the evaluation target, so its reported
near-perfect reconstruction was invalid. The corrected implementation and evaluation pass
only the per-block transition offset/value summary to the reconstructor.

| Task | ZOH | spline+satfix | B-spline+satfix | gripper_sync |
|---|---:|---:|---:|---:|
| RC-CloseSingleDoor | 100.000% | 100.000% | 100.000% | 100.000% |
| RC-CoffeePressButton | 97.559% | 97.559% | 97.559% | 100.000% |
| RC-TurnOffMicrowave | 99.657% | 99.657% | 99.657% | 100.000% |
| RC-TurnOffSinkFaucet | 100.000% | 100.000% | 100.000% | 100.000% |

These results are reproducible with `python eval_gripper_sync_openloop.py --k 4`; the
machine-readable output is `result_gripper_sync_openloop.json`.

## Closed-loop RoboCasa, k=4, preliminary, n=15

The existing baseline outcomes are shown below. A second bridge attempt after correcting
the oracle bug failed during `socket.socket(...)` with `PermissionError: [Errno 1]
Operation not permitted`. The current execution interface does not expose a
`danger-full-access` mode, so `robocasa_bridge.py` still cannot start in this sandbox.
Missing results remain explicit rather than being inferred from the open-loop metric.

| Task | native | ZOH | spline+satfix | B-spline+satfix | gripper_sync |
|---|---:|---:|---:|---:|---:|
| RC-OpenDrawer | 40.0% (6/15) | 26.7% (4/15) | 26.7% (4/15) | pending | pending |
| RC-PnPCounterToStove | 6.7% (1/15) | 6.7% (1/15) | 6.7% (1/15) | pending | pending |

Exact paired McNemar p-values for `gripper_sync` versus each baseline are pending the same
run. For context only, the existing OpenDrawer native-versus-ZOH contrast is exact p=0.5
(2 native-only, 0 ZOH-only), while all existing PnPCounterToStove contrasts are p=1.0.
No significance claim is made from these preliminary n=15 cells.

Run the missing paired evaluations, reusing the existing checkpoints and without retraining,
with the bridge in `robocasa_uv` running in a separate terminal:

```bash
# Bridge terminal
/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  robocasa_bridge.py --port 8765

# Evaluation terminal
/home/user/anaconda3/envs/gr00t/bin/python harness.py RC-OpenDrawer \
  --steps 15000 --n_train 39 --n_eval 15 --k 4 \
  --arms native,zoh,spline_satfix,bspline_eps_satfix,gripper_sync \
  --suffix _gripper_sync

/home/user/anaconda3/envs/gr00t/bin/python harness.py RC-PnPCounterToStove \
  --steps 15000 --n_train 39 --n_eval 15 --k 4 \
  --arms native,zoh,spline_satfix,bspline_eps_satfix,gripper_sync \
  --suffix _gripper_sync
```

## Scope

This is a first result on the correct benchmark, not yet a Holm-corrected pre-registered
claim. The next step for a Q2 submission is to pre-register a larger-n confirmatory run on
this same mechanism.
