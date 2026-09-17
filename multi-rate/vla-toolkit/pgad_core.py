"""Shared code for the Progress-Gated Adaptive Denoising (PGAD) toolkit:
- a corrected progress-head trainer (normalizes BOTH image features and
  state, unlike vla-rft/progress_estimator.py's train_head which only
  normalizes state -- see train_progress_heads.log: libero_goal's head
  diverged to a degenerate sigmoid-saturated constant, train_mse frozen at
  0.3347 for 250+ epochs, held-out r=-0.48. Root cause: raw SigLIP-pooled
  features (1920-dim) are not zero-mean/unit-std, so an unlucky checkpoint's
  feature scale can push the pre-sigmoid logit into a saturated, zero-gradient
  region at lr=1e-3 with no grad clipping. Fixing this at the source (feature
  normalization + grad clipping) rather than only patching the one suite that
  visibly failed, so all 4 suites use one uniform, documented recipe.)
- the ProgressGate: turns a per-decision-point progress estimate into a
  noise_level choice for the flow-SDE sampler (the actual PGAD mechanism).
"""
from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/Desktop/multi-rate/vla-rft")
from progress_estimator import ProgressHead, embed_batch  # noqa: E402 -- reused unmodified
from flow_sde import sample_actions_flow_sde  # noqa: E402 -- reused unmodified
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402

DEVICE = "cuda"

# PGAD gate constants (fixed, pre-registered before any eval run -- NOT fit to
# eval data). MARGIN uses a running max rather than the immediately-previous
# point to absorb normal regressor jitter (r~0.98, not 1.0) without treating
# noise as a stall.
MARGIN = 0.05          # progress must drop >5pp below its running peak to count as "stalled"
NOISE_DEFAULT = 0.1    # matches this project's existing default (flow_sde.py, all prior phases)
NOISE_CAREFUL = 0.02   # ponytail: fixed 5x reduction, not tuned; a swept schedule is the upgrade
                        # path if this coarse two-level gate under/over-triggers.


def train_head_normalized(data, epochs=300, lr=1e-3, grad_clip=1.0):
    feat_mean = data["train"]["feats"].mean(axis=0)
    feat_std = data["train"]["feats"].std(axis=0) + 1e-6
    state_mean = data["train"]["states"].mean(axis=0)
    state_std = data["train"]["states"].std(axis=0) + 1e-6

    def make_xy(split):
        feats = (data[split]["feats"] - feat_mean) / feat_std
        states = (data[split]["states"] - state_mean) / state_std
        x = np.concatenate([feats, states], axis=1)
        y = data[split]["progress"]
        return torch.tensor(x, dtype=torch.float32, device=DEVICE), torch.tensor(y, dtype=torch.float32, device=DEVICE)

    x_train, y_train = make_xy("train")
    x_val, y_val = make_xy("val")

    head = ProgressHead(in_dim=x_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        pred = head(x_train)
        loss = nn.functional.mse_loss(pred, y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            head.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(head(x_val), y_val).item()
            print(f"  epoch {ep}: train_mse={loss.item():.4f} val_mse={val_loss:.4f}", flush=True)
    train_time = time.time() - t0

    head.eval()
    with torch.no_grad():
        val_pred = head(x_val).cpu().numpy()
    val_true = y_val.cpu().numpy()
    overall_corr = float(np.corrcoef(val_pred, val_true)[0, 1])

    per_ep_corrs = []
    for ep in np.unique(data["val"]["ep"]):
        mask = data["val"]["ep"] == ep
        if mask.sum() >= 3 and val_pred[mask].std() > 1e-8:
            per_ep_corrs.append(float(np.corrcoef(val_pred[mask], val_true[mask])[0, 1]))

    stats = {
        "train_time_s": train_time,
        "n_train_frames": len(y_train),
        "n_val_frames": len(y_val),
        "n_train_episodes": data["n_train_ep"],
        "n_val_episodes": data["n_val_ep"],
        "held_out_overall_pearson_r": overall_corr,
        "held_out_per_episode_pearson_r_mean": float(np.mean(per_ep_corrs)) if per_ep_corrs else None,
        "held_out_per_episode_pearson_r_n": len(per_ep_corrs),
        "held_out_val_mse": float(np.mean((val_pred - val_true) ** 2)),
    }
    return head, feat_mean, feat_std, state_mean, state_std, stats


def predict_progress(head, feat_mean, feat_std, state_mean, state_std, feats, states):
    feats_n = (feats - feat_mean) / feat_std
    states_n = (states - state_mean) / state_std
    x = np.concatenate([feats_n, states_n], axis=1)
    with torch.no_grad():
        return head(torch.tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy()


def save_head(path, head, feat_mean, feat_std, state_mean, state_std):
    # store normalization stats as tensors (not numpy) so the checkpoint is
    # loadable with weights_only=True -- these files are only ever written by
    # train_progress_heads.py in this same folder, but there's no reason to
    # need the unsafe unpickle path for a plain dict of tensors.
    torch.save({
        "state_dict": head.state_dict(),
        "feat_mean": torch.as_tensor(feat_mean), "feat_std": torch.as_tensor(feat_std),
        "state_mean": torch.as_tensor(state_mean), "state_std": torch.as_tensor(state_std),
    }, path)


def load_head(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    feat_mean, feat_std = ckpt["feat_mean"].cpu().numpy(), ckpt["feat_std"].cpu().numpy()
    state_mean, state_std = ckpt["state_mean"].cpu().numpy(), ckpt["state_std"].cpu().numpy()
    head = ProgressHead(in_dim=len(feat_mean) + len(state_mean)).to(DEVICE)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, feat_mean, feat_std, state_mean, state_std


class ProgressGate:
    """Stateful per-episode gate: feed it the current frame, get back the
    noise_level to use for the NEXT action chunk. mode='gated' is the PGAD
    method; 'always_careful' and 'off' are the ablation arms (off == the
    existing frozen baseline's sampling, i.e. always NOISE_DEFAULT)."""

    def __init__(self, head, feat_mean, feat_std, state_mean, state_std, resize_wh, mode="gated",
                 stall_noise_level=None):
        assert mode in ("gated", "always_careful", "off")
        self.head, self.feat_mean, self.feat_std = head, feat_mean, feat_std
        self.state_mean, self.state_std = state_mean, state_std
        self.resize_wh = resize_wh
        self.mode = mode
        # overridable stall-response noise level -- lets a pilot test the
        # opposite-sign hypothesis (raise noise to escape a stall instead of
        # lowering it) without duplicating this class. Defaults to the
        # original NOISE_CAREFUL for both 'gated' and 'always_careful'.
        self.stall_noise_level = stall_noise_level if stall_noise_level is not None else NOISE_CAREFUL
        self.p_max = None
        self.history = []

    def reset(self):
        self.p_max = None
        self.history = []

    def observe_and_decide(self, policy, img1, img2, state):
        """img1/img2: (3,H,W) CPU float tensors [0,1]. state: (8,) np array.
        Returns (progress_value, noise_level, stalled_bool)."""
        if self.mode == "off":
            return None, NOISE_DEFAULT, False
        if self.mode == "always_careful":
            return None, self.stall_noise_level, True

        feats = embed_batch(policy, [img1], [img2], self.resize_wh, batch_size=1)
        p = float(predict_progress(
            self.head, self.feat_mean, self.feat_std, self.state_mean, self.state_std,
            feats, state[None, :].astype(np.float32),
        )[0])
        self.p_max = p if self.p_max is None else max(self.p_max, p)
        stalled = p < self.p_max - MARGIN
        self.history.append(p)
        return p, (self.stall_noise_level if stalled else NOISE_DEFAULT), stalled


@torch.no_grad()
def rollout_episode_pgad(policy, preproc, env, task_description, seed, screen, max_steps, gate):
    """Same flow-SDE replanning mechanics as progress_estimator.collect_rollout /
    multi_task_campaign.rollout_episode, but the noise_level for each new action
    chunk comes from `gate` instead of a fixed constant. gate.mode='off' exactly
    reproduces the existing frozen-baseline sampling (NOISE_DEFAULT every chunk)."""
    gate.reset()
    policy.eval()
    policy._queues[ACTION] = deque([], maxlen=policy.config.n_action_steps)
    for key in list(policy._queues.keys()):
        if key != ACTION:
            policy._queues[key] = deque([], maxlen=policy.config.n_action_steps)

    observation, _ = env.reset(seed=seed)
    resize_wh = policy.config.resize_imgs_with_padding
    success = False
    step = 0
    decision_log = []
    for step in range(1, max_steps + 1):
        if len(policy._queues[ACTION]) == 0:
            raw = screen._policy_input(observation, task_description)
            img1 = raw["observation.images.image"].squeeze(0).cpu()
            img2 = raw["observation.images.wrist_image"].squeeze(0).cpu()
            state_np = raw["observation.state"].squeeze(0).numpy()

            progress, noise_level, stalled = gate.observe_and_decide(policy, img1, img2, state_np)
            decision_log.append({"progress": progress, "noise_level": noise_level, "stalled": stalled})

            batch = preproc(raw)
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
            proc_batch = policy._prepare_batch(dict(batch))
            policy._queues = populate_queues(policy._queues, proc_batch, exclude_keys=[ACTION])
            for k in proc_batch:
                if k in policy._queues and k != ACTION:
                    proc_batch[k] = torch.stack(list(policy._queues[k]), dim=1)

            images, img_masks = policy.prepare_images(proc_batch)
            state = policy.prepare_state(proc_batch)
            lang_tokens = proc_batch["observation.language.tokens"]
            lang_masks = proc_batch["observation.language.attention_mask"]

            actions_shape = (1, policy.model.config.chunk_size, policy.model.config.max_action_dim)
            noise = policy.model.sample_noise(actions_shape, DEVICE)
            result = sample_actions_flow_sde(
                policy, images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, noise_level=noise_level, stochastic=True,
            )
            actions = result["actions"]
            original_action_dim = policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            policy._queues[ACTION].extend(actions.transpose(0, 1)[: policy.config.n_action_steps])

        action = policy._queues[ACTION].popleft()
        obs_action = action.unsqueeze(0) if action.dim() == 1 else action
        env_action = policy._rl_postprocessor(obs_action).to("cpu").float().numpy().reshape(-1)
        observation, _, terminated, truncated, info = env.step(env_action)
        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    n_stalled = sum(1 for d in decision_log if d["stalled"])
    return {
        "success": success, "steps": step,
        "n_decisions": len(decision_log), "n_stalled_decisions": n_stalled,
        "progress_trace": [d["progress"] for d in decision_log],
    }


# --- PGRF: Progress-Gated Replan Frequency (fresh mechanism, replaces the
# killed noise-gating idea) --------------------------------------------------
# Same progress signal, same frozen heads, same stall test (p < running-max -
# MARGIN) -- but instead of touching the sampler's noise_level, it controls
# HOW MANY actions from a freshly-sampled chunk get executed before the next
# replan: STALLED_HORIZON=1 (current default, maximally responsive) when
# struggling, ON_TRACK_HORIZON=5 (fewer VLM forward passes -> cheaper) when
# progressing normally. chunk_size=50 for this checkpoint (native training
# horizon), so committing to 5 steps is a small fraction of what the model was
# actually trained to predict at once -- not a stretch.
ON_TRACK_HORIZON = 5
STALLED_HORIZON = 1


class ReplanGate:
    """Mirrors ProgressGate's stall detection but decides a replan HORIZON
    (# actions to commit to) instead of a noise_level."""

    def __init__(self, head, feat_mean, feat_std, state_mean, state_std, resize_wh, mode="gated"):
        assert mode in ("gated", "off")  # 'off' == always STALLED_HORIZON (== existing frozen baseline)
        self.head, self.feat_mean, self.feat_std = head, feat_mean, feat_std
        self.state_mean, self.state_std = state_mean, state_std
        self.resize_wh = resize_wh
        self.mode = mode
        self.p_max = None

    def reset(self):
        self.p_max = None

    def observe_and_decide(self, policy, img1, img2, state):
        if self.mode == "off":
            return None, STALLED_HORIZON, False
        feats = embed_batch(policy, [img1], [img2], self.resize_wh, batch_size=1)
        p = float(predict_progress(
            self.head, self.feat_mean, self.feat_std, self.state_mean, self.state_std,
            feats, state[None, :].astype(np.float32),
        )[0])
        self.p_max = p if self.p_max is None else max(self.p_max, p)
        stalled = p < self.p_max - MARGIN
        return p, (STALLED_HORIZON if stalled else ON_TRACK_HORIZON), stalled


@torch.no_grad()
def rollout_episode_pgrf(policy, preproc, env, task_description, seed, screen, max_steps, gate):
    """Same mechanics as rollout_episode_pgad, but the gate's decision sets how
    many actions from the freshly-sampled chunk get queued (replan horizon),
    not the sampler's noise_level (always NOISE_DEFAULT here)."""
    gate.reset()
    policy.eval()
    policy._queues[ACTION] = deque([], maxlen=policy.model.config.chunk_size)
    for key in list(policy._queues.keys()):
        if key != ACTION:
            policy._queues[key] = deque([], maxlen=policy.config.n_action_steps)

    observation, _ = env.reset(seed=seed)
    success = False
    step = 0
    decision_log = []
    n_replans = 0
    for step in range(1, max_steps + 1):
        if len(policy._queues[ACTION]) == 0:
            raw = screen._policy_input(observation, task_description)
            img1 = raw["observation.images.image"].squeeze(0).cpu()
            img2 = raw["observation.images.wrist_image"].squeeze(0).cpu()
            state_np = raw["observation.state"].squeeze(0).numpy()

            progress, horizon, stalled = gate.observe_and_decide(policy, img1, img2, state_np)
            decision_log.append({"progress": progress, "horizon": horizon, "stalled": stalled})
            n_replans += 1

            batch = preproc(raw)
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
            proc_batch = policy._prepare_batch(dict(batch))
            policy._queues = populate_queues(policy._queues, proc_batch, exclude_keys=[ACTION])
            for k in proc_batch:
                if k in policy._queues and k != ACTION:
                    proc_batch[k] = torch.stack(list(policy._queues[k]), dim=1)

            images, img_masks = policy.prepare_images(proc_batch)
            state = policy.prepare_state(proc_batch)
            lang_tokens = proc_batch["observation.language.tokens"]
            lang_masks = proc_batch["observation.language.attention_mask"]

            actions_shape = (1, policy.model.config.chunk_size, policy.model.config.max_action_dim)
            noise = policy.model.sample_noise(actions_shape, DEVICE)
            result = sample_actions_flow_sde(
                policy, images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, noise_level=NOISE_DEFAULT, stochastic=True,
            )
            actions = result["actions"]
            original_action_dim = policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            policy._queues[ACTION].extend(actions.transpose(0, 1)[:horizon])  # <-- the gate's effect

        action = policy._queues[ACTION].popleft()
        obs_action = action.unsqueeze(0) if action.dim() == 1 else action
        env_action = policy._rl_postprocessor(obs_action).to("cpu").float().numpy().reshape(-1)
        observation, _, terminated, truncated, info = env.step(env_action)
        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    n_stalled = sum(1 for d in decision_log if d["stalled"])
    return {
        "success": success, "steps": step, "n_replans": n_replans,
        "n_decisions": len(decision_log), "n_stalled_decisions": n_stalled,
        "progress_trace": [d["progress"] for d in decision_log],
    }
