# VLA Multi-Rate Benchmark Tracker

_Converted from `vla_multirate_benchmark_tracker.xlsx` (frozen snapshot as of 2026-09-05; superseded by `multi-rate/full_grid_2026-09-07/` for anything after that date)._


## Overview & Methods

VLA Multi-Rate Benchmark Tracker: DeepONet-v2, Flow Matching, ARC & Splines

Master evaluation scoreboard comparing Continuous Action-Rate Conditioning (ARC) vs. Post-hoc Spline Resamplers (Cubic & B-Spline) and Parameter-Free Baselines.


**1. Method Architecture & Deployment Matrix**


| Method Name | Model Family | Rate Conditioning (ARC) | Resampling / Interpolator | Action Space Calibration | Checkpoints Status | Primary Finding |
|---|---|---|---|---|---|---|
| DeepONet v2 + ARC (ASRC) | DeepONet (Continuous) | Yes (Trunk Fourier Projection) | Learned Operator Re-querying | Magnitude Scaling + Cadence | Available (8.3k & 30k on all suites) | Recovers off-native rate to 71.3% (tied with splines/ZOH) |
| DeepONet v2 + Cubic Spline | DeepONet (Discrete Chunk) | No (Fixed 20Hz Head) | Cubic Spline (C2 Continuous) | Magnitude Scaling + Cadence | Available (8.3k & 30k on all suites) | Recovers 40Hz to 73.3% on Spatial; matches operator |
| DeepONet v2 + B-Spline | DeepONet (Discrete Chunk) | No (Fixed 20Hz Head) | B-Spline / Basis Smoothing | Magnitude Scaling + Cadence | Available (Evaluated on Spatial & ManiSkill) | Smooth trajectories; equivalent to cubic spline under power |
| Flow Matching + ARC | Flow Matching (Continuous) | Yes (Rate Vector Field Embedding) | Continuous Denoising Drift | Integrated Normalization | MISSING (Never implemented/trained) | Architecture open gap (estimated ~10h training needed) |
| Flow Matching + Cubic Spline | Flow Matching (Standard) | No (20Hz Chunk Generation) | Cubic Spline (C2 Continuous) | Magnitude Scaling + Cadence | Available (8.3k & 30k on all suites) | SOTA in-distribution & 40Hz recovery (78.0%–94.0%) |
| Flow Matching + B-Spline | Flow Matching (Standard) | No (20Hz Chunk Generation) | B-Spline Resampling | Magnitude Scaling + Cadence | Available (Evaluated via LeRobot API) | Matches Cubic Spline; robust to inference staleness |

Related assets: tri-policy video at `multi-rate/vla_tri_policy_closed_loop_comparison.mp4`, all-tasks 10/20/40Hz video at `multi-rate/vla_tri_policy_all_tasks_10_20_40hz.mp4`, JSON results at `multi-rate/tri_policy_all_tasks_10_20_40hz_results.json`.


## LIBERO (4 Suites)

LIBERO Benchmark (Spatial, Object, Goal, 10 / Long) — Closed-Loop Evaluations

Matched Budget Evaluation: 8,300 Steps (1,650 Head Warmup + 6,650 Fine-Tune, Batch Size 48, 11.0s Horizon, 0.5s Replan)


| Benchmark Suite | Control Rate | Method & Head | Arm / Configuration | Episodes (N) | Success Count | Success Rate | Delta vs Native | Status / Notes |
|---|---|---|---|---|---|---|---|---|
| LIBERO-Spatial | 20 Hz (Native) | DeepONet v2 + ARC | asrc_native_20env (Base) | 150 | 114 | 0.76 | 0 | Native Reference (8.3k) |
| LIBERO-Spatial | 20 Hz (Native) | DeepONet v2 (Plain) | til_native_20env (Base) | 150 | 101 | 0.6733 | 0 | Native Reference (til 8.3k) |
| LIBERO-Spatial | 20 Hz (Native) | Flow Matching | flow8300_native_20env | 50 | 40 | 0.8 | 0 | Native Reference (Flow 8.3k) |
| LIBERO-Spatial | 40 Hz | DeepONet v2 + ARC | asrc_cadmag_folding_40env | 150 | 107 | 0.7133 | -0.0467 | Operator Folding + CadMag |
| LIBERO-Spatial | 40 Hz | DeepONet v2 + Cubic Spline | asrc_cadmag_spline_40env | 150 | 110 | 0.7333 | -0.0267 | Cubic Spline + CadMag |
| LIBERO-Spatial | 40 Hz | DeepONet v2 + B-Spline | asrc_cadmag_bspline_40env | 50 | 36 | 0.72 | -0.04 | B-Spline + CadMag |
| LIBERO-Spatial | 40 Hz | Flow + Cubic Spline | flow8300_cadmag_spline_40env | 50 | 39 | 0.78 | -0.02 | Flow Spline + CadMag (SOTA) |
| LIBERO-Spatial | 40 Hz | Flow + B-Spline | flow8300_cadmag_bspline_40env | 50 | 38 | 0.76 | -0.04 | Flow B-Spline + CadMag |
| LIBERO-Spatial | 40 Hz | All Models (Naive) | naive_unadapted_40env | 50 | 2 | 0.04 | -0.76 | Severe Rate Collapse (0-4%) |
| LIBERO-Spatial | 10 Hz | DeepONet v2 + ARC | asrc_mag_folding_10env | 50 | 31 | 0.62 | -0.14 | Operator Fold + Magscale |
| LIBERO-Spatial | 10 Hz | Flow + Cubic Spline | flow8300_mag_spline_10env | 50 | 39 | 0.78 | -0.02 | Flow Spline + Magscale |
| LIBERO-Spatial | 10 Hz | Flow + B-Spline | flow8300_mag_bspline_10env | 50 | 37 | 0.74 | -0.06 | Flow B-Spline + Magscale |
| LIBERO-Spatial | 10 Hz | DeepONet v2 (10Hz Finetune) | asrc10hz_ft_s0 (Replan=5) | 50 | 34 | 0.68 | -0.08 | Narrowed consistency loss |
| LIBERO-Object | 20 Hz (Native) | DeepONet v2 + ARC | asrc_object_s0 | 50 | 40 | 0.8 | 0 | Native Reference (8.3k) |
| LIBERO-Object | 20 Hz (Native) | Flow Matching | flow8300_object_s0 | 50 | 45 | 0.9 | 0 | Flow SOTA (+10.0 pp win) |
| LIBERO-Object | 40 Hz | DeepONet v2 + ARC | asrc_object_cadmag_folding | 40 | 32 | 0.8 | 0 | Pilot run verified |
| LIBERO-Object | 40 Hz | DeepONet v2 + Cubic Spline | asrc_object_cadmag_spline | 40 | 33 | 0.825 | 0.025 | Pilot run verified |
| LIBERO-Object | 40 Hz | Flow + Cubic Spline | flow8300_object_spline_40env | 50 | 44 | 0.88 | -0.02 | Calculated from cadence grid |
| LIBERO-Goal | 20 Hz (Native) | DeepONet v2 + ARC | asrc_goal_s0 | 50 | 45 | 0.9 | 0 | Native Reference (8.3k) |
| LIBERO-Goal | 20 Hz (Native) | Flow Matching | flow8300_goal_s0 | 50 | 47 | 0.94 | 0 | Flow SOTA (+4.0 pp win) |
| LIBERO-Goal | 40 Hz | DeepONet v2 + ARC | asrc_goal_cadmag_folding | 40 | 35 | 0.875 | -0.025 | Pilot run verified |
| LIBERO-Goal | 40 Hz | Flow + Cubic Spline | flow8300_goal_spline_40env | 50 | 45 | 0.9 | -0.04 | Calculated from cadence grid |
| LIBERO-10 (Long) | 20 Hz (Native) | DeepONet v2 + ARC | asrc_long_s0 | 50 | 26 | 0.52 | 0 | Native Reference (8.3k) |
| LIBERO-10 (Long) | 20 Hz (Native) | Flow Matching | flow8300_10_s0 | 50 | 31 | 0.62 | 0 | Flow SOTA (+10.0 pp win) |
| LIBERO-10 (Long) | 40 Hz | DeepONet v2 + ARC | asrc_long_cadmag_folding | 40 | 21 | 0.525 | 0.005 | Pilot run verified |
| LIBERO-10 (Long) | 40 Hz | Flow + Cubic Spline | flow8300_10_spline_40env | 50 | 30 | 0.6 | -0.02 | Calculated from cadence grid |


## Plus, ManiSkill & MetaWorld

Robustness & Cross-Benchmark Evaluations (LIBERO-Plus, ManiSkill 3, MetaWorld MT50)


**1. LIBERO-Plus (Robustness & OOD Disturbance)**


| Benchmark | Control Rate | Method / Interpolator | Trials (N) | Success Count | Success Rate | Delta vs. Spline | Significance |
|---|---|---|---|---|---|---|---|
| LIBERO-Plus | 10 Hz | Operator Fold (ARC) | 20 | 3 | 0.15 | 0.1 | Exploratory Pilot |
| LIBERO-Plus | 10 Hz | Cubic Spline | 20 | 1 | 0.05 | 0 | Exploratory Pilot |
| LIBERO-Plus | 20 / 40 Hz | Operator Fold (ARC) | 315 | 200 | 0.6349 | 0.006 | p = 0.8555 (Tied) |
| LIBERO-Plus | 20 / 40 Hz | Cubic Spline | 315 | 198 | 0.6286 | 0 | Baseline Reference |
| LIBERO-Plus | 20 / 40 Hz | B-Spline | 315 | 197 | 0.6254 | -0.0032 | p = 0.9100 (Tied) |


**2. ManiSkill 3 (Action Chunk Decimation Grid, n=900 rollouts/cell)**


| Task Name | Decimation (Rate) | Original Reference | Exact ZOH (Integral) | Cubic Spline | B-Spline / PCHIP | TAC-Fold (Operator) | TAC vs ZOH (p-value) |
|---|---|---|---|---|---|---|---|
| LiftPegUpright-v1 | k=2 (20 Hz) | 0.9733 | 0.4678 | 0.1556 | 0.2867 | 0.4889 | +2.11 pp (p=0.204 Tied) |
| PickCube-v1 | k=2 (20 Hz) | 0.9256 | 0.57 | 0.4522 | 0.4156 | 0.73 | +16.00 pp (p=2.1e-21 Win) |
| PickCube-v1 | k=3 (~13 Hz) | 0.9256 | 0.2956 | 0.1278 | 0.1844 | 0.3367 | +4.11 pp (p=0.019 Win) |
| PickCube-v1 | k=4 (10 Hz) | 0.9244 | 0.2078 | 0.0578 | 0.1111 | 0.1544 | -5.33 pp (p=2.6e-4 Loss) |
| PickCube-v1 | k=5 (8 Hz) | 0.9256 | 0.2833 | 0.0256 | 0.0833 | 0.1067 | -17.67 pp (p=2.3e-25 Loss) |
| PushCube-v1 | k=2 (20 Hz) | 0.9978 | 0.9856 | 0.9356 | 0.9689 | 0.9667 | -1.89 pp (p=2.2e-4 Loss) |
| PushCube-v1 | k=3 (~13 Hz) | 0.9789 | 0.9256 | 0.8744 | 0.91 | 0.88 | -4.56 pp (p=1.3e-8 Loss) |
| PushCube-v1 | k=4 (10 Hz) | 0.9856 | 0.9056 | 0.7667 | 0.8611 | 0.77 | -13.56 pp (p=4.2e-20 Loss) |
| PushCube-v1 | k=5 (8 Hz) | 0.9867 | 0.7544 | 0.7022 | 0.6344 | 0.6767 | -7.78 pp (p=8.6e-12 Loss) |


**3. MetaWorld MT50 (Pre-Registered 6,000 Closed-Loop Episodes across 50 Envs)**


| Benchmark | Control Rate | RateFold (Operator) | Zero-Order Hold (ZOH) | PCHIP / Spline | Delta vs. ZOH | Exact McNemar p | Holm-Corrected p |
|---|---|---|---|---|---|---|---|
| MetaWorld MT50 | 5 Hz | 0.294 | 0.332 | 0.284 | -0.038 | p = 0.053 | p = 0.371 (Not Sig) |
| MetaWorld MT50 | 10 Hz | 0.43 | 0.434 | 0.428 | -0.004 | p = 0.896 | p = 1.000 (Not Sig) |
| MetaWorld MT50 | 30 Hz | 0.498 | 0.522 | 0.492 | -0.024 | p = 0.112 | Not Sig |
| MetaWorld MT50 | 40 Hz | 0.488 | 0.516 | 0.48 | -0.028 | p = 0.084 | Not Sig |


## Checkpoint Inventory

Master Checkpoint Inventory & Training Requirements Tracker


| Architecture | Benchmark Suite | Steps / Budget | Status | File Size | Local / Target File Path | Training Time Needed |
|---|---|---|---|---|---|---|
| DeepONet + ARC (asrc_s0) | LIBERO-Spatial | 8,300 | AVAILABLE | 904.5 MB | .../claim_campaign_20260804/runs/asrc_s0/checkpoints/8300 | 0.0 hrs (Ready) |
| DeepONet + ARC (asrc_object_s0) | LIBERO-Object | 8,300 | AVAILABLE | 904.5 MB | .../claim_campaign_20260804/runs/asrc_object_s0/checkpoints/8300 | 0.0 hrs (Ready) |
| DeepONet + ARC (asrc_goal_s0) | LIBERO-Goal | 8,300 | AVAILABLE | 904.5 MB | .../claim_campaign_20260804/runs/asrc_goal_s0/checkpoints/8300 | 0.0 hrs (Ready) |
| DeepONet + ARC (asrc_long_s0) | LIBERO-10 (Long) | 8,300 | AVAILABLE | 904.5 MB | .../claim_campaign_20260804/runs/asrc_long_s0/checkpoints/8300 | 0.0 hrs (Ready) |
| DeepONet + ARC (asrc30k_s0) | LIBERO-Spatial | 30,000 | AVAILABLE | 904.5 MB | .../claim_campaign_20260804/runs/asrc30k_s0/checkpoints/30000 | 0.0 hrs (Ready) |
| DeepONet v2 Plain (v2_spatial) | LIBERO-Spatial | 30,000 | AVAILABLE | 904.3 MB | .../v2/deeponet_results/Spatial/runs/m3_s0/checkpoints/30000 | 0.0 hrs (Ready) |
| DeepONet v2 Plain (v2_object) | LIBERO-Object | 30,000 | AVAILABLE | 904.3 MB | .../v2/deeponet_results/Object/runs/m3_s0/checkpoints/30000 | 0.0 hrs (Ready) |
| DeepONet v2 Plain (v2_goal) | LIBERO-Goal | 30,000 | AVAILABLE | 904.3 MB | .../Goal/runs/m3_s0/checkpoints/30000 | 0.0 hrs (Ready) |
| DeepONet v2 Plain (v2_long) | LIBERO-10 (Long) | 30,000 | AVAILABLE | 904.3 MB | .../v2/deeponet_results/Long/runs/m3_s0/checkpoints/30000 | 0.0 hrs (Ready) |
| Flow Matching (flow8300 Spatial) | LIBERO-Spatial | 8,300 | AVAILABLE | 864.7 MB | .../runs/m1_flow_s0/checkpoints/8300 | 0.0 hrs (Ready) |
| Flow Matching (flow_s0 Spatial) | LIBERO-Spatial | 30,000 | AVAILABLE | 864.7 MB | .../paper_repro/Spatial/runs/flow_s0/checkpoints/30000 | 0.0 hrs (Ready) |
| Flow Matching (flow_s0 Object) | LIBERO-Object | 30,000 | AVAILABLE | 864.7 MB | .../paper_repro/Object/runs/flow_s0/checkpoints/30000 | 0.0 hrs (Ready) |
| Flow Matching (flow_s0 Long) | LIBERO-10 (Long) | 30,000 | AVAILABLE | 864.7 MB | .../paper_repro/Long/runs/flow_s0/checkpoints/30000 | 0.0 hrs (Ready) |
| Flow Matching (flow8300 Goal) | LIBERO-Goal | 8,300 | REMOTE NODE | 864.7 MB | blackwell:/home/user/flow8300_goal_s0 | 0.0 hrs (Fetch via SCP) |
| Flow Matching + ARC | LIBERO-Spatial | 8,300 | MISSING | — | runs/flow_arc_spatial_s0 | ~2.4 hrs on Blackwell |
| Flow Matching + ARC | LIBERO-Object | 8,300 | MISSING | — | runs/flow_arc_object_s0 | ~2.7 hrs on Blackwell |
| Flow Matching + ARC | LIBERO-Goal | 8,300 | MISSING | — | runs/flow_arc_goal_s0 | ~2.4 hrs on Blackwell |
| Flow Matching + ARC | LIBERO-10 (Long) | 8,300 | MISSING | — | runs/flow_arc_long_s0 | ~2.5 hrs on Blackwell |
| DeepONet v2 (Native 10Hz Retrain) | LIBERO-Spatial | 8,300 | MISSING | — | runs/deeponet_10hz_s0 | ~2.5 hrs on Blackwell |
| DeepONet v2 (Native 40Hz Retrain) | LIBERO-Spatial | 8,300 | MISSING | — | runs/deeponet_40hz_s0 | ~2.5 hrs on Blackwell |
| Flow Matching (Native 10Hz Retrain) | LIBERO-Spatial | 8,300 | MISSING | — | runs/flow_10hz_s0 | ~2.5 hrs on Blackwell |
| Flow Matching (Native 40Hz Retrain) | LIBERO-Spatial | 8,300 | MISSING | — | runs/flow_40hz_s0 | ~2.5 hrs on Blackwell |
