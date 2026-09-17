# Phase 1 Status: Flow-SDE RL Port on SmolVLA — Verification, Port, Smoke Test

No training run started. Read-only on `multi-rate/vla-vault` (its `output/`/`wiki`/ledgers are untouched).
New code lives entirely in `/home/user/Desktop/multi-rate/vla-rft/`.

## Task 1 — Checkpoint architecture (blocking question)

**TASK:** Determine whether `flow8300_s0/checkpoints/8300` was trained with stock
`lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy` or a project-specific
variant (`modeling_smolvla_ph.py` / `modeling_smolvla_deeponet.py`).

**COMMAND:**
```
grep -n "head" "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/train.py"
```
**SOURCE:** `train.py:56-68` (`build_policy`) — `--head flow` imports
`from modeling_smolvla_ph import SmolVLAPHPolicy` with `ph_enabled=(variant=="baseline")→False`.

**RESULT:** The checkpoint WAS built via `SmolVLAPHPolicy`, not the stock class directly —
but `modeling_smolvla_ph.py:31` states in a docstring comment *"ph_enabled=False makes this
class behave EXACTLY like vanilla SmolVLA"* / *"byte-identical to vanilla [SmolVLA]"*, and the
class subclasses stock `SmolVLAPolicy`/`VLAFlowMatching` (`modeling_smolvla_ph.py:43-49`),
overriding only the training-loss `forward()` (adds a PH surrogate loss when `ph_enabled=True`)
— it does **not** override `denoise_step`/`sample_actions`. Since `flow8300_s0` was trained
with `variant=baseline` → `ph_enabled=False`, the saved weights are architecturally identical
to stock SmolVLA.

**CHECK (executed, passed):**
```
conda run -n vla_smolvla_libero python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
pol = SmolVLAPolicy.from_pretrained('/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300')
print('LOADED OK, type:', type(pol).__name__)
print('num params:', sum(p.numel() for p in pol.parameters()))"
```
Output: `LOADED OK, type: SmolVLAPolicy` / `num params: 450046176`, 489/489 weight tensors
loaded with no missing/unexpected keys.

**VERDICT:** Confirmed — stock `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy` loads
this checkpoint cleanly. Also confirms 450M param count matches the vault's `flow8300_spatial`
arm (79.40%±1.56%) cited in `PROJECT_STATUS_REPORT_2026-08-28.md` §4.2.

---

## Task 2 — Locate the Flow-SDE mechanism source

**TASK:** Find `sample_mean_var`/`get_timesteps`/`gaussian_logprob` (referenced by import,
not opened, in the earlier feasibility check's scratch copy of `rl_action_model.py`).

**SOURCE:** Not found as a standalone `rl_sampler.py` on disk or via a fresh RLinf clone —
instead, the **same math is already fully inlined** (not just imported) in the earlier
feasibility check's saved scratch copy of RLinf's PI0 action-model class:
`/tmp/claude-1000/.../scratchpad/openpi_action_model.py`, methods
`_get_timesteps` (L1120), `sample_mean_var_val` (L1125, the `flow_sde` branch at L1176-1183),
`get_velocity` (L1248), `get_logprob_norm` (L1287). This is a self-contained reference
implementation — no external package needed.

**RESULT:** Ported directly (see Task 3). **UNKNOWN / not verified:** whether this file is
byte-identical to the current upstream `RLinf/RLinf` GitHub repo (did not re-clone to diff,
since the inline implementation was sufficient and already read verbatim by a prior agent) —
if exact upstream parity matters later (e.g. for a paper's methods section claiming "we use
π_RL's public implementation"), diff against a fresh clone before writing that sentence.

---

## Task 3 — Port (`flow_sde.py`)

**TASK:** Implement single-random-step Flow-SDE sampling + log-prob, hooked into SmolVLA's
`denoise_step`/`embed_prefix`.

**SOURCE:** `/home/user/Desktop/multi-rate/vla-rft/flow_sde.py` (new file, ~140 lines).

**Algebraic verification performed (by hand, not just assumed):** RLinf's `flow_ode` branch
of `sample_mean_var_val` (deterministic case, `x_t_std=0`) reduces exactly to
`x_t_mean = x_t + v_t * (t_next - t) = x_t - delta * v_t`, which is identical to SmolVLA's
native `x_t = x_t + dt * v_t` (`dt = -1/num_steps = -delta`) in
`modeling_smolvla.py:865`. Also confirmed `RLinf`'s `timesteps = linspace(1, 1/N, N)` produces
the exact same sequence as SmolVLA's `time = 1.0 + step*dt` for `step in range(N)`, and `delta`
is constant `= 1/N` at every step (not just first/last). This justifies mixing RLinf's
`flow_sde` branch (one step) with SmolVLA's native stepping (all other steps) in one loop.

**CHECK:** see Task 4 smoke test — the `flow_sde` branch's output is what's exercised there.

---

## Task 4 — Smoke test

**TASK:** Load checkpoint, run Flow-SDE sampler on a small batch, confirm finite log-probs,
correct action shape, and gradient flow via `.backward()`.

**COMMAND:** `conda run -n vla_smolvla_libero python smoke_test.py`

**RESULT (first attempt — FAILED, informative):** All gradients were exactly `0.0` despite
`log_prob.requires_grad == True`. Root cause (diagnosed, not guessed): the sampled action
`x_t_next = x_t_mean + eps*x_t_std` was passed *undetached* into
`gaussian_logprob(x_t_next, x_t_mean, x_t_std)`. Because `x_t_mean` appears both inside
`x_t_next`'s construction and as the subtracted mean in the log-prob, the two paths cancel
**exactly** in the backward pass (`d/d(x_t_mean)[(x_t_mean + eps·x_t_std) − x_t_mean] ≡ 0`) —
a real algebraic identity, not a numerical artifact. **Fix:** detach `x_t_next` before scoring
it, matching the correct RL pattern (sample now during rollout as a fixed target; recompute its
log-prob under the current — possibly updated — policy later). This is exactly the
rollout/update separation any GRPO/PPO loop needs regardless, so the fix is not a workaround,
it's the correct design going into Task 3 of the implementation plan (the GRPO loop).

**RESULT (after fix — PASSED):**
```
CHECK actions.shape: (2, 50, 32)
CHECK chosen_idx: 7 / 10
CHECK log_prob finite: True
CHECK log_prob shape: (2, 50, 32)
CHECK log_prob requires_grad: True
CHECK params_with_grad: 155 / 500
CHECK max grad norm: 52035.0546875
CHECK any nonzero grad: True

SMOKE TEST PASSED
```
(155/500 params receiving gradients is expected, not a bug — `freeze_vision_encoder: true` in
the checkpoint's `config.json` freezes a large fraction of the network.)

**Note for Phase 2 (not yet acted on):** max grad norm (~52k) is large for an un-clipped single
backward pass through ~7 chained denoising steps (chosen_idx was 7/10, so gradients flow back
through 7 sequential model forward calls). The checkpoint's own `config.json` already specifies
`optimizer_grad_clip_norm: 10.0`, which should be reused unchanged in the GRPO optimizer step —
flagging this now so it isn't rediscovered the hard way during the first real training run.
Also: steps *after* `chosen_idx` don't need gradient tracking (only `log_prob` at `chosen_idx`
is used for the RL loss) — Phase 2's rollout code should wrap those in `torch.no_grad()` for
memory/speed, unlike this smoke test which left them differentiable for simplicity.

---

## Files created

- `/home/user/Desktop/multi-rate/vla-rft/flow_sde.py`
- `/home/user/Desktop/multi-rate/vla-rft/smoke_test.py`
- `/home/user/Desktop/multi-rate/vla-rft/PHASE1_STATUS.md` (this file)

## Explicitly unresolved (zero-faking: not claiming these)

- `rl_sampler.py`'s exact standalone location / upstream RLinf repo parity — not verified (Task 2).
- Whether `gate_prereg_SEALED.json` formally defines "Gate 0" — not opened this phase either
  (out of scope for Phase 1; relevant again once we report a result against that gate).
- No LIBERO-Spatial *environment* rollout was run this phase — the smoke test used dummy
  random-tensor observations shaped like the checkpoint's `input_features`, not real sim
  frames. Task 6 of the plan (3-seed baseline re-eval) will be the first real-environment run.
