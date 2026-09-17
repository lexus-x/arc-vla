"""Flow-SDE single-random-step stochastic sampler for SmolVLA.

Ports the mechanism from RLinf's openpi_action_model.py (sample_mean_var_val /
get_logprob_norm, github.com/RLinf/RLinf), which implements pi_RL's Flow-SDE
trick (arXiv:2510.25889) for making RL log-likelihoods tractable on a flow-
matching action head: only ONE randomly-chosen denoise step is made stochastic
(Gaussian, mean/std derived below); every other step stays the model's native
deterministic Euler update.

The deterministic ("flow_ode") algebra used here is verified (by hand, see
PHASE1_STATUS.md) to be exactly equivalent to SmolVLA's native
`x_t = x_t + dt * v_t` Euler step (dt = -1/num_steps), so mixing them in a
single loop is safe.

Reference source read verbatim at:
/tmp/claude-1000/-home-user-Desktop/a4b0f32d-.../scratchpad/openpi_action_model.py
lines ~1120-1300 (RLinf's PI0RLActionModel).
"""

from __future__ import annotations

import math
import random

import torch
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def get_delta(num_steps: int) -> float:
    """Constant step size; algebraically equals SmolVLA's |dt| = 1/num_steps."""
    return 1.0 / num_steps


def sample_mean_std_sde(x_t: torch.Tensor, v_t: torch.Tensor, t_input: float, delta: float, noise_level: float):
    """flow_sde branch of RLinf's sample_mean_var_val, for one step at time t_input, step size delta."""
    device, dtype = x_t.device, x_t.dtype
    t = torch.tensor(t_input, device=device, dtype=dtype)
    d = torch.tensor(delta, device=device, dtype=dtype)
    # avoid 1/(1-1)=inf at the very first denoise step (t=1.0)
    denom_t = t if float(t.item()) != 1.0 else (t - d)
    sigma_ratio = t / (1.0 - denom_t)
    sigma_i = noise_level * torch.sqrt(sigma_ratio.clamp(min=0.0))

    x0_pred = x_t - v_t * t
    x1_pred = x_t + v_t * (1.0 - t)
    x0_weight = 1.0 - (t - d)
    x1_weight = (t - d) - sigma_i**2 * d / (2.0 * t)
    x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
    x_t_std = (torch.sqrt(d) * sigma_i).expand_as(x_t_mean)
    return x_t_mean, x_t_std


def gaussian_logprob(sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Ported from RLinf's get_logprob_norm (non-safe branch)."""
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    constant_term = -torch.log(sigma_safe) - 0.5 * math.log(2 * math.pi)
    exponent_term = -0.5 * ((sample - mu) / sigma_safe) ** 2
    log_prob = constant_term + exponent_term
    return torch.where(mask, torch.zeros_like(log_prob), log_prob)


def sample_actions_flow_sde(
    policy,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise: torch.Tensor | None = None,
    noise_level: float = 0.1,
    stochastic: bool = True,
    forced_idx: int | None = None,
    replay_target: torch.Tensor | None = None,
):
    """Reimplementation of VLAFlowMatching.sample_actions with ONE randomly-chosen
    denoise step made stochastic (flow-SDE), matching pi_RL's mechanism.

    Returns dict with: actions (x_0), log_prob (at the chosen step, zeros elsewhere
    conceptually but we only return the chosen step's log_prob), chosen_idx.

    replay_target: GRPO update-time mode. When given (a stored, already-sampled
    x_t_next from a prior ROLLOUT call at the same forced_idx, same noise), skip
    drawing a fresh eps -- score `replay_target` against freshly-recomputed
    mean/std under the CURRENT policy params instead. This is the standard
    "act now under old params, reweight under new params" RL update: mean/std
    still depend on `model.denoise_step` -> gradients flow correctly, but the
    sampled action itself is held fixed to what was actually executed in the
    environment. `noise` MUST be the exact tensor used at rollout time (stored
    and replayed byte-for-byte) so the pre-chosen-step trajectory matches.
    """
    model = policy.model
    bsize = state.shape[0]
    device = state.device

    if noise is None:
        actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
        noise = model.sample_noise(actions_shape, device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )

    num_steps = model.config.num_steps
    delta = get_delta(num_steps)
    chosen_idx = forced_idx if forced_idx is not None else random.randint(0, num_steps - 1)

    x_t = noise
    log_prob_chosen = None
    chosen_step_sample = None  # the intermediate x_t_next AT chosen_idx (NOT final x_0) --
    # this, not the fully-denoised "actions", is what a replay call must be scored against.
    for step in range(num_steps):
        t_input = 1.0 - step * delta  # matches SmolVLA's `time = 1.0 + step*dt`, dt=-delta
        time_tensor = torch.tensor(t_input, dtype=torch.float32, device=device).expand(bsize)

        # Only the chosen step needs gradient (mean/std there depend on v_t, which
        # depends on current params via denoise_step + the grad-enabled prefix
        # encoding above). Steps before it are context, not the RL decision -- run
        # them detached. Steps after it are unneeded when replaying for an update
        # (we only need log_prob_chosen), so we early-exit instead of computing them.
        step_grad_enabled = replay_target is not None and step == chosen_idx
        step_ctx = torch.enable_grad() if step_grad_enabled else torch.no_grad()
        with step_ctx:
            v_t = model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )

            if stochastic and step == chosen_idx:
                x_t_mean, x_t_std = sample_mean_std_sde(x_t, v_t, t_input, delta, noise_level)
                if replay_target is not None:
                    # Update-time: score the STORED rollout's intermediate sample (at
                    # this same chosen step) against freshly recomputed mean/std (which
                    # carries gradient via v_t / denoise_step under current params).
                    # Do not draw new randomness here.
                    x_t_next = replay_target
                    log_prob_chosen = gaussian_logprob(x_t_next, x_t_mean, x_t_std)
                    chosen_step_sample = x_t_next
                    break  # x_0 not needed in replay mode; stop as soon as we have it
                else:
                    eps = torch.randn_like(x_t_mean)
                    x_t_next = (x_t_mean + eps * x_t_std).detach()  # rollout sample: a fixed target,
                    # NOT differentiated through — mirrors RL practice (act now, reweight the log-prob
                    # of that fixed action under the *current* params later). Computing log_prob from the
                    # same (undetached) mean/std tensors that produced the sample gives an exact-zero
                    # gradient (mean cancels algebraically: d/dmean[(mean+eps*std)-mean] == 0), which is
                    # why this detach matters, not merely a style choice.
                    log_prob_chosen = gaussian_logprob(x_t_next, x_t_mean, x_t_std)
                    chosen_step_sample = x_t_next
                    continue
            # deterministic Euler step, algebraically == flow_ode branch == SmolVLA native step
            x_t_next = x_t + (-delta) * v_t

        x_t = x_t_next

    return {
        "actions": x_t,  # final x_0 (rollout mode); mid-trajectory / unused (replay mode)
        "chosen_step_sample": chosen_step_sample,  # the intermediate sample AT chosen_idx --
        # store THIS (not "actions") as replay_target for a later update call.
        "log_prob": log_prob_chosen,
        "chosen_idx": chosen_idx,
        "num_steps": num_steps,
    }
