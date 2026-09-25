# Pre-registration (SEALED) — learned-shape governor

Drafted 2026-09-23 20:35 KST; SEALED 2026-09-23 21:32 KST on the user's approval ("yes", 21:29 KST),
before any confirmation episode was run (no `result_*_confirm*.json` existed). Code frozen at
sealing (sha256):
```
6b186d315ad6aa8405296959734e22f4370a4a6099bc505d837d7c423a38dc73  shape_governor.py
20cca670c5209f103cd547133e040306ea2f8457990e1a4d6df2614996152042  harness.py
0dbd49da4943a2631a44457d8e9e0e161bff7a4156488b479dc93ab5f6cc37a5  resample_qp.py
d8cdc067c265f6797e06ec076dc671eb8e2063a9028b63239cc5a4548d5e1faa  resample_math.py
```
Result files carry the suffix `_confirm`. Any later amendment is appended below with a timestamp
and must precede the results it affects.

## What is already known (dev only, NOT citable)
PickCube-v1, DP, k=4, eval window eligible[600:700] (n=100), paired:
qp_learned 65, learned_raw 66, tac_fold_satfix 52, qp_anchor 52, arc+qp_anchor 44, native 82.
qp_learned vs tac_fold_satfix +13pp (18/5, p=0.011, uncorrected, one of 9+ dev tests).
k=2 dev: every arm 82-85 (ceiling). This dev window picked the method; it is never reused.

## Method (frozen)
`shape_governor.py` as of this draft: MLP 2x256 GELU, input = 2-frame raw obs + block means of
the executed 8-step chunk, output = zero-sum-per-block residual; AdamW 1e-3, wd 1e-4, 4000 steps,
batch 256, seed 0, 10% of training demos held out for the fit report only. Trained per (task, k)
on the SAME 200 training demos as the DP policy. Anchor -> `resample_qp_anchor` (exact block
sums, |v|<=1). No hyperparameter may change after the first confirmation episode.

## Policies, tasks, windows (frozen)
- DP checkpoints: `dp_<task>.pt`, 30k steps, 200 demos, already trained (no retraining).
- Tasks (8): PickCube, RollBall, PullCube, LiftPegUpright, PushCube, AnymalC-Reach, PokeCube,
  StackCube (ManiSkill RL demos, pd_joint_delta_pos).
- Windows: 7 new tasks `--eval-offset 100 --n_eval 400` (eligible[300:700], never evaluated;
  eligible[200] was touched by the training-only run). PickCube `--eval-offset 500 --n_eval 293`
  (eligible[700:993], all that remains unused).
- k = 4 (primary). k = 2 secondary, run only after all k=4 cells finish.

## Arms (frozen)
native, zoh, tac_fold_satfix, qp_anchor, qp_learned, learned_raw, learned_tanh, mlp_bc.
`learned_tanh` = the same MLP recipe trained to output the whole chunk through tanh (bounded,
no QP, block totals not enforced) — the "why not just bound the network?" alternative.
`mlp_bc` = the same MLP recipe predicting the full 8-step chunk (all action dims) from the 2-frame
obs alone, executed as the policy (no DP). Answers "why not just use the small network?".

## Hypotheses and tests
Exact two-sided McNemar per comparison. A comparison is VOID if both arms are <10% or >90%.

- **H1 (primary, Holm m=8):** qp_learned > tac_fold_satfix at k=4, one test per task.
  Claim "qp_learned beats the best post-hoc resampler" requires Holm-significant wins on >=4 of
  8 tasks AND no Holm-significant loss on any task. Otherwise the claim is reported as failed.
- **H2 (secondary, Holm m=8):** qp_learned > qp_anchor at k=4.
- **H3 (secondary, no direction predicted):** qp_learned vs mlp_bc. If mlp_bc >= qp_learned on
  most tasks, the paper must say the small network alone explains the gain.
- **H4 (secondary, directional, Holm m=4):** qp_learned > learned_raw on the tasks where the
  learned anchor exceeds ±1 on >=15% of values (measured 2026-09-23 on held-out TRAINING demos,
  `qp_role_offline_k4.json`, before any confirmation episode): RollBall 30.3%, LiftPegUpright
  23.0%, PullCube 20.3%, AnymalC-Reach 16.9%. Predicted null where it is low (PickCube 6.6%,
  consistent with the dev tie 65 vs 66). Offline, clipping loses 0.008-0.050 of commanded motion
  per block; the QP loses 0 by construction.
- **H6 (secondary, Holm m=8):** qp_learned > learned_tanh at k=4. Offline, the tanh net loses
  4-13x more commanded motion per block than clipping and has the highest error on all 8 tasks.
- **H5 (exploratory):** correlation of per-task (qp_learned − tac_fold_satfix) with the demo
  saturation fraction measured before this draft.
- k=2: same H1 family, reported separately, Holm m=8.

## Not allowed after results
No change to windows, arms, recipe, tasks, checkpoints, or the >=4/8 rule. No dropping tasks or
k=2. Losses and voids reported as measured.

## Planned separately (each needs its own amendment before running)
External baseline (RTC-style inpainting or low-pass), Flow Matching replication, real-VLA cell.

## Amendment 1 — 2026-09-23 21:58 KST (execution environment only; no protocol change; before any result)
Tasks are split across two identical-GPU machines; every arm of a task runs on ONE machine, so
pairing is unaffected. Local (blackwell): RollBall, LiftPegUpright, PullCube, AnymalC-Reach.
blackwell2 (WSL2, same RTX PRO 6000 Blackwell, torch 2.13.0+cu130, mani_skill 3.0.1, sapien 3.0.3):
PushCube, PokeCube, StackCube, PickCube. WSL2 has no NVIDIA Vulkan, so on blackwell2 ManiSkill gets
render_backend="none" via a sitecustomize shim plus Mesa lavapipe (CPU Vulkan) for material
creation. obs_mode=state: rendering never feeds observations or physics. harness.py,
shape_governor.py, resample_*.py are byte-identical (sha256 above, verified on blackwell2).
Shape/tanh/bc models for all 8 tasks were fitted on blackwell with the frozen recipe
(prefit_models.py, 6 CPU threads) and copied; AnymalC's tanh/bc were fitted by its own harness run.
