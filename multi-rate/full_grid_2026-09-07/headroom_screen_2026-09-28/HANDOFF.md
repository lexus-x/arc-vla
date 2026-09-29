# Handoff: 4-benchmark-family headroom screen (2026-09-28 17:30 KST)

Goal: positive Q2 method paper, where the learned governor beats the converters on at least 4 simulation benchmark **families**.
The user ruled out policy training (time), so only official pretrained checkpoints are used.
Plan: `/home/user/.claude/plans/status-on-our-proposal-parallel-yao.md`. Rule: `SCREEN_RULE.md` (frozen, hashes in `SCREEN_RULE.sha256`, amendment 1 = LIBERO).

## Screen results (dev seeds 5000+, n=50 per arm, native vs ZOH only)
| Family | Checkpoint | Verdict |
|---|---|---|
| ManiSkill3 | our DP (sealed campaign) | pass (5/8 Holm wins already) |
| Push-T | lerobot/diffusion_pusht (`hub_diffusion_pusht`) | ZOH screen passed at k=2 (66/52), but **converter check (amendment 3) FAILS**: native 66, best converter 62 (tac_fold_satfix) at k=2 (gap 4pp), 58 (qp_anchor) at k=4 (gap 8pp) — both under the 10pp bar. `pusht_conv_k2.json`, `pusht_conv_k4.json`. **Push-T is not a Q2 candidate.** |
| Meta-World MT10 | lerobot/smolvla_metaworld | **fail**: 0/10 tasks with ≥10 pp at k=8/4/2 (max +8) |
| ALOHA transfer-cube | lerobot/act_aloha_sim_transfer_cube_human, migrated to `hub_act_aloha_transfer_cube` | **fail**: native 80; ZOH 82/78/82 at k=2/4/8 |
| RoboMimic can/square/transport/tool_hang (ph) | official DP `.ckpt` | ZOH screen passed at k=4 (marginal), but **converter check (amendment 3) FAILS**: can 100→98 (2pp), square 94→88 (6pp), transport 80→92 (−12pp), tool_hang 90→76 (14pp, OPEN). Only 1/4 open, below the ≥half bar. `dp_*_{spline_satfix,tac_fold_satfix,qp_anchor}_k4.json`. Note: all 3 converters gave numerically-identical outputs on can_ph/square_ph (verified not a bug — n_action_steps=8 means only 2 coarse blocks at k=4, a degenerate case where spline/TAC-Fold/QP-anchor's constrained solutions coincide; confirmed they diverge normally at 4+ blocks). tool_hang_ph differed because its satfix saturation step reacts differently near clip bounds. **RoboMimic is not a Q2 candidate** — k=2 and k=8 are unavailable, so no other k to retry. |
| Block Pushing | official DP-T | **cannot pass**: n_action_steps = 1, so no valid k |
| LIBERO (4 suites) | lerobot/smolvla_libero | **pass at k=8** (3/4 suites ≥10 pp): native/ZOH-k8 libero_10 60/44, object 70/52, spatial 74/62, goal 70/64. k=2 fails 0/4, k=4 fails 1/4 (libero_10 60/50). From `libero_*/eval_info.json` `overall.pc_success`. |
| Franka Kitchen | official DP-C `kitchen/.../epoch=1700-test_mean_score=0.580.ckpt` (amendment 4, 4a) | downloaded + smoked. Ckpt is `abs_action=true` (Robot_PosAct). Native p4 = 100% (verified real: 48/50 episodes complete exactly 4 subtasks, mean 4.04, matches the ckpt's own reported test score 0.580×7≈4.06 — near-zero-variance policy, not a bug). **Converter check FAILS at both k=2 (gap 0pp: native 100, spline_satfix 100) and k=4 (gap ≤2pp: native 100, spline/tac_fold 98, qp_anchor pending but can only raise best_conv further).** **Kitchen is not a Q2 candidate.** |
| RoboTwin 2.0 | lerobot/smolvla_robotwin | 5th family (insurance for the marginal RoboMimic pass). Screened under amendment 2 (launched 20:11 KST, detached) by `run_robotwin_screen.sh` → `rt_*`, stage lines + VERDICT in `run_robotwin_screen.out`. Rerunning the script resumes: finished cells are skipped. Smoke (burned seeds 9900+): native 1/2, ~50 s/episode solo. Under load (P=10) the GPU saturates at ~11 steps/s total (~1.1/s per cell): native stage ≲7 h (open_microwave's 1500-step limit is the tail), full screen possibly 10–20 h. |

**Converter check partial result:** RoboMimic can_ph (already excluded — didn't pass ZOH at k=4) closed as expected: native 100%, spline_satfix 98%, tac_fold_satfix 98% (gap 2pp).

**LIBERO converter check finished (2026-09-29 01:52 KST, `tabulate_conv.py`): LIBERO is not a Q2 candidate.** Only 1/4 suites open at k=8 (libero_object +12pp); libero_10 closed to +8pp, libero_spatial +6pp, and libero_goal went *negative* (native 70, best converter 72 — qp_anchor beat native). The 21:31 "hopeful" partial reading (3/12 cells, all showing 14-16pp gaps) did not hold once the remaining 9 cells finished. **All 4 families that passed the ZOH screen (Push-T, RoboMimic, Kitchen, LIBERO) have now failed the converter check.** VLABench (added as a 6th candidate after the 2026-09-28 jobs were unstuck) also failed: `lerobot/smolvla_vlabench` scored 0/14 across 5 diverse tasks in the pilot — see SCREEN_RULE.md amendment 5. **Only ManiSkill has a real win. RoboTwin (relaunched 01:52 KST, 3/10 native cells already done) is the only screen still in flight.**

**Converter check (amendment 3, launched 21:12):** a win must beat spline_satfix, tac_fold_satfix and qp_anchor, so each eligible family also runs those arms (Push-T k=2,4 → `pusht_conv_k*.json`; RoboMimic k=4 → `dp_*_{arm}_k4.json` via `run_dp_screen.sh conv`; LIBERO k=8 → `libero_*_{arm}_k8/` via `run_libero_screen.sh conv`). A family stays a candidate only if native − best converter ≥ 10 pp on ≥ half its tasks.

**Goal (user, 2026-09-28 20:47): Holm-significant method wins in ≥5 sim benchmark families.** Wins so far: 1 (ManiSkill, sealed campaign). The screen above only selects *eligible* families (ZOH loses something the method can recover): Push-T, RoboMimic (marginal) and LIBERO are eligible, RoboTwin is pending. The method has not been run on any of them yet.

Detached jobs (these survive the session closing): `run_libero_screen.sh`, the `eval_dp_official_rate.py` xargs, and `retry_libero.sh` (after the first LIBERO pass it reruns cells that failed, e.g. CUDA OOM at 17:27, at P=2; the final count goes to `run_libero_screen.out`).

## Tools written today
- `../eval_lerobot_rate.py`: wraps lerobot_eval. Env vars: RATE_K, RATE_ARM, RATE_HOLD (delta mode) or RATE_ABS=1 + RATE_HOLD_IDX (absolute targets). Paired noise per chunk call. Self-check passes.
- RoboTwin env `robotwin`: keep numpy 1.26.4 (sapien/mplib). The lerobot `[dataset]` extra was installed package by package, because `pip install -e lerobot` pins numpy>=2. `eval_lerobot_rate.py` with `--env.type=robotwin` skips UnStableError seeds (s+100000·j) and swaps the OIDN denoiser for OptiX: bundled OIDN 2.0.1 has no sm_120, so frames came out grainy. It also turns off lerobot's 10 hardcoded video episodes (render() re-renders all 3 RT cameras, ~17% of GPU time). Relaunched 20:21 KST with videos off. bw2 (WSL2: SAPIEN finds no GPU render device, only CPU lavapipe without ray tracing) and a100 (no ssh key; A100 has no RT cores) cannot run the RT renderer.
- The Meta-World / ALOHA / LIBERO runner scripts are in this directory (`run_*_screen.sh`). Meta-World needs MUJOCO_GL=egl, empty_cameras=2, and a rename map (see the script).

## Next steps
1. Screen done: 3 eligible families besides ManiSkill (table above). Eligible ≠ win.
2. RoboTwin: read the VERDICT in `run_robotwin_screen.out`. If the runner died, rerun `setsid nohup ./run_robotwin_screen.sh >> run_robotwin_screen.out 2>&1 &` (it resumes).
3. For passing families: adapt `shape_governor.py`. It currently assumes T=8 delta chunks with a [-1, 1] box. The new families need T = chunk length, and Push-T/ALOHA need absolute targets via the displacement path. Then fit the governor (a small MLP, ~1 min, not policy training) on each family's demo dataset.
4. Put the passing families, per-family k, tasks and seeds into `../PREREG_Q2_PROPRIO_GOVERNOR_DRAFT.md`, hash, and seal with user approval before any confirmatory episode.
