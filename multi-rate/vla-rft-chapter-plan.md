# VLA-RFT Chapter Plan: Flow-SDE RL Post-Training on SmolVLA (LIBERO-Spatial)

Reconnaissance only — no training run started. New chapter within `multi-rate/vla-vault`, reusing existing infra. Read-only on the vault itself; this file lives outside `output/`/`wiki/`.

## What already exists (verified paths)

- **Conda env**: `vla_smolvla_libero` — Python 3.12.13, torch 2.10.0+cu128 (CUDA 12.8, `torch.cuda.is_available()==True` on this Blackwell GPU, driver CUDA 13.0 — compatible), lerobot 0.5.1 editable at `/home/user/lerobot`, robosuite 1.4.0, `hf_libero` 0.1.3. No flash-attn installed — not required for this port (SmolVLA/PaliGemma path doesn't need it here).
- **SmolVLA flow-matching code**: `/home/user/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` — `sample_actions` (L801, Euler loop, `num_steps = self.config.num_steps`, `dt = -1.0/num_steps`), `denoise_step` (L872), training loss `forward` (L355/L766). No separate `flow_matching.py`; corrects the earlier feasibility-check note — logic is inline in this one file. (GR00T's separate `flow_matching_action_head.py` is unrelated, different policy.)
- **Existing training script**: `/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/train.py`, invoked via `run_flow8300_suites_train.sh`. Confirmed recipe: `--head flow --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999`, base ckpt `lerobot/smolvla_base`.
- **Existing LIBERO-Spatial flow-matching SFT checkpoint** (reuse this, skip SFT-from-scratch): `/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300/` (also `1650/`, `LATEST.txt`). This is the exact `flow8300_spatial` arm cited in the project status report (79.40%±1.56% in-dist).
- **Eval harness**: `/home/user/eval_reg_s0_libero.py` is a working, shardable LIBERO-Spatial eval loop (uses `libero_plus_wrapper.LiberoPlusEnv`, `evaluate_plus.plus_obs_to_policy_input`, `SmolVLADeepONetPolicy.from_pretrained`) — adapt this directly for RL-checkpoint eval rather than writing a new harness.
- **RL mechanism reference**: RLinf's `rl_action_model.py` (scratch copy read) shows the actual Flow-SDE trick is smaller than expected: pick **one** random denoise step index uniformly, run only that step stochastically (Gaussian, mean/std from `rl_sampler.sample_mean_var(x_t, v_t, idx, noise_method, noise_level, num_steps)`), log-prob via `rl_sampler.gaussian_logprob`; all other steps stay deterministic Euler. Needs porting `rl_sampler.sample_mean_var`/`get_timesteps`/`gaussian_logprob` (not yet located as standalone file — only referenced, not opened; **UNKNOWN**, locate in RLinf source before implementing) and hooking at SmolVLA's `denoise_step`/`sample_actions`.

## Gate 0 — NOT located

Searched `war_room_config.yaml`, `output/gate_2026-08-28/*` (5 files present: `gate_gate_lp3.json`, `gate_gate_lp5.json`, `gate_gate_unfiltered.json`, `gate_prereg_SEALED.json`, a host-idle log) and grepped the vault for "Gate 0" — no definition file found, only the number (≥87% Spatial, 500 trials, 3 seeds) as prose in `PROJECT_STATUS_REPORT_2026-08-28.md`. **UNKNOWN**: whether `gate_prereg_SEALED.json` defines this gate — not opened, flagging for the main session to check before citing a formal gate spec.

## Eval-harness traps that apply directly (from `wiki/failures/eval-harness-traps.md`)

- **Budget confound**: comparing an RL-post-trained checkpoint against the 8,300-step SFT baseline is fine (same base), but do not compare against any 30K-step arm — that's worth ~+26pp on its own per this vault's own finding.
- **Head/checkpoint mismatch**: assert loaded checkpoint's action-head tensor shapes match expected before running — this project has been burned by silent random-reinit before.
- **`init_state_id` drift bias**: pin eval seeds explicitly per trial; do not let success/failure change which init states get seen.
- **Per-category noise floor**: ~10.5pp drift on reruns — a 1-seed RL result must be framed as preliminary per the already-agreed scope, not as a significant claim.

## Ordered implementation steps (estimate)

1. Locate `rl_sampler.py` (or equivalent) in the RLinf repo clone/pip package for the exact `sample_mean_var`/`gaussian_logprob`/`get_timesteps` implementations (~0.5 day).
2. Port those 3 functions + a `_predict_train`-style single-stochastic-step sampler into a new module hooking SmolVLA's `denoise_step`/`sample_actions` (~1 day).
3. Write a minimal GRPO loop: batched rollout via the eval harness pattern above but with `mode="train"`, sparse LIBERO success/failure reward, group-relative advantage (no value head/critic) (~1.5 days).
4. Load `flow8300_s0/checkpoints/8300` as the RL starting point (no SFT redo needed) (~0 extra days — reuse).
5. 3-seed SFT baseline re-eval on LIBERO-Spatial using `eval_reg_s0_libero.py`'s pattern (adapt `--ckpt` to the flow checkpoint) to get a fresh noise-floor stddev before/alongside RL training (~0.5–1 day, can run in parallel with steps 1–3 on a second GPU).
6. 1-seed RL training + eval run, debugging pass included (~1.5–2 days).

Total: ~5–6 days to a first result, consistent with the earlier feasibility estimate; leaves buffer inside the 1–1.5 week fallback scope.

## Not verified / explicitly unknown

- Exact location of `rl_sampler.py` — not opened.
- Whether `gate_prereg_SEALED.json` formally defines "Gate 0" — not opened.
- Whether `train.py`'s `--head flow` path exactly matches stock `lerobot` SmolVLA's `modeling_smolvla.py` or a modified copy (`modeling_smolvla_deeponet.py`/`modeling_smolvla_ph.py` exist alongside it in the same dir — the RL port should target stock `lerobot.policies.smolvla.modeling_smolvla`, not these project-specific variants, to keep the new chapter decoupled from the DeepONet experiment code; confirm which the `flow8300_s0` checkpoint was actually trained with before loading it).
