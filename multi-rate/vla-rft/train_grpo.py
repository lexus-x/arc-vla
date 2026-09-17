"""Minimal GRPO training loop: Flow-SDE RL post-training of SmolVLA on LIBERO-Spatial.

Pilot scope (per agreed fallback): 2 tasks (reusing evaluate_height_screen.py's
pinned TASK_IDS = (3, 5), a pre-existing diagnostic scope in this vault -- not
a shortcut invented here), single inner epoch (rollout under current params,
immediately compute the update from those same rollouts -- no importance
sampling / multiple PPO epochs, which keeps this correct without needing a
clipped ratio), group-relative advantage (GRPO), sparse terminal success/fail
reward, no critic/value head, grad_clip_norm=10.0 (checkpoint's own config).

Rollout mechanics replicate SmolVLAPolicy.select_action's queue management
(see modeling_smolvla.py:322-347) but route chunk generation through
flow_sde.sample_actions_flow_sde instead of the native deterministic sampler,
and run under torch.no_grad() (rollout only needs the sampled action's value,
not its gradient -- the gradient is recomputed at update time via
replay_target, see flow_sde.py's docstring on that parameter).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, CAMPAIGN)

import suite_screen  # noqa: E402  -- generalized replacement for evaluate_height_screen
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402

from flow_sde import sample_actions_flow_sde  # noqa: E402

DEVICE = "cuda"
# `screen` is set per-suite in main() via `global screen; screen = suite_screen.get_screen(args.suite)`
# before any rollout runs -- module functions below resolve it at call time, not def time.
screen = None
NOISE_LEVEL = 0.1
GRAD_CLIP_NORM = 10.0  # from the checkpoint's own config.json (optimizer_grad_clip_norm)
LR = 1e-6  # conservative: 10x below the checkpoint's own backbone SFT lr (1e-5); RL updates
           # are higher-variance (single stochastic step, sparse reward) so a smaller LR keeps
           # this pilot from destabilizing the SFT policy in a handful of updates.


@torch.no_grad()
def rollout_episode(policy, preprocessor, env, task_description: str, seed: int, group_id: str):
    """One LIBERO episode using the Flow-SDE stochastic sampler for chunk generation.
    Returns dict: success, steps, transitions (list of replayable decision points)."""
    policy.eval()
    policy._queues[ACTION] = deque([], maxlen=policy.config.n_action_steps)
    for key in list(policy._queues.keys()):
        if key != ACTION:
            policy._queues[key] = deque([], maxlen=policy.config.n_action_steps)

    observation, _ = env.reset(seed=seed)
    transitions = []
    success = False
    step = 0
    for step in range(1, screen.MAX_STEPS + 1):
        if len(policy._queues[ACTION]) == 0:
            # Match evaluate_height_screen._rollout exactly: raw obs -> preprocessor
            # pipeline (normalization/tokenization) -> policy's own queue bookkeeping.
            batch = preprocessor(screen._policy_input(observation, task_description))
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
                noise=noise, noise_level=NOISE_LEVEL, stochastic=True,
            )
            actions = result["actions"]
            original_action_dim = policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            transitions.append({
                "images": [im.detach().cpu() for im in images],
                "img_masks": [m.detach().cpu() for m in img_masks],
                "lang_tokens": lang_tokens.detach().cpu(),
                "lang_masks": lang_masks.detach().cpu(),
                "state": state.detach().cpu(),
                "noise": noise.detach().cpu(),
                "chosen_idx": result["chosen_idx"],
                # the intermediate sample AT the chosen step, NOT the final denoised
                # x_0 ("actions") -- log_prob is scored at that single step, so replay
                # must target the same intermediate value, in the same pre-unpad
                # (full max_action_dim) space the denoising loop operates in.
                "sampled_action": result["chosen_step_sample"].detach().cpu(),
            })
            policy._queues[ACTION].extend(actions.transpose(0, 1)[: policy.config.n_action_steps])

        action = policy._queues[ACTION].popleft()
        obs_action = action.unsqueeze(0) if action.dim() == 1 else action
        # postprocess via the policy's stats-based unnormalizer, matching _rollout's pattern
        env_action = policy_postprocess(policy, obs_action)
        observation, _, terminated, truncated, info = env.step(env_action)
        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    return {"success": success, "steps": step, "transitions": transitions, "group_id": group_id}


def policy_postprocess(policy, action_tensor):
    # `select_action` normally returns already-postprocessed actions via the
    # pipeline's postprocessor; here we replicate that by calling the stored
    # postprocessor directly (attached by main() as policy._rl_postprocessor).
    action = policy._rl_postprocessor(action_tensor)
    return action.to("cpu").float().numpy().reshape(-1)


def compute_update(policy, transitions_with_advantage, optimizer):
    """One gradient step over a batch of (transition, advantage) pairs, GRPO-style:
    loss = -mean_i[ advantage_i * log_prob_i ], log_prob_i recomputed under current
    params via replay_target (see flow_sde.sample_actions_flow_sde docstring)."""
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    n = 0
    for tr, adv in transitions_with_advantage:
        if adv == 0.0:
            continue  # degenerate group (all-same reward): zero advantage, skip (standard GRPO)
        images = [im.to(DEVICE) for im in tr["images"]]
        img_masks = [m.to(DEVICE) for m in tr["img_masks"]]
        lang_tokens = tr["lang_tokens"].to(DEVICE)
        lang_masks = tr["lang_masks"].to(DEVICE)
        state = tr["state"].to(DEVICE)
        noise = tr["noise"].to(DEVICE)
        target = tr["sampled_action"].to(DEVICE)

        result = sample_actions_flow_sde(
            policy, images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, noise_level=NOISE_LEVEL, stochastic=True,
            forced_idx=tr["chosen_idx"], replay_target=target,
        )
        log_prob = result["log_prob"]  # (1, chunk, action_dim)
        loss = -(adv * log_prob.mean())
        loss.backward()
        total_loss += loss.item()
        n += 1

    if n == 0:
        optimizer.zero_grad(set_to_none=True)
        return {"n_transitions": 0, "loss": None, "grad_norm": None}

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], GRAD_CLIP_NORM
    )
    optimizer.step()
    return {"n_transitions": n, "loss": total_loss / n, "grad_norm": float(grad_norm)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--n_groups_per_update", type=int, default=2)  # 2 tasks x 4 rollouts = 8/update
    parser.add_argument("--n_updates", type=int, default=15)
    parser.add_argument("--out_ckpt", default="/home/user/Desktop/multi-rate/vla-rft/rl_checkpoint")
    parser.add_argument("--log_json", default="/home/user/Desktop/multi-rate/vla-rft/logs/train_progress.json")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--ckpt", default="/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300")
    args = parser.parse_args()

    global screen
    screen = suite_screen.get_screen(args.suite)

    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preprocessor, postprocessor) = screen.load_policy("flow", args.ckpt, stats)
    policy._rl_postprocessor = postprocessor
    policy.train()
    for p in policy.parameters():
        p.requires_grad_(p.requires_grad)  # keep checkpoint's own frozen/unfrozen split

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=LR
    )

    progress = []
    task_envs = {}
    for tid in screen.TASK_IDS:
        env = screen._make_env(tid)
        task_envs[tid] = env

    ep_counter = 0
    t0 = time.time()
    for update_idx in range(args.n_updates):
        batch_transitions = []
        update_rewards = []
        for tid in screen.TASK_IDS:
            env = task_envs[tid]
            group_id = f"u{update_idx}_t{tid}"
            group_rewards = []
            group_transitions_per_ep = []
            for g in range(args.group_size):
                seed = 1000 + ep_counter
                ep_counter += 1
                env.init_state_id = g % max(1, screen.TRIALS_PER_TASK)
                out = rollout_episode(policy, preprocessor, env, env.task_description, seed=seed, group_id=group_id)
                reward = 1.0 if out["success"] else 0.0
                group_rewards.append(reward)
                group_transitions_per_ep.append(out["transitions"])
                print(f"[update {update_idx}] task={tid} rollout={g} success={out['success']} "
                      f"steps={out['steps']} n_chunks={len(out['transitions'])}", flush=True)

            mean_r = float(np.mean(group_rewards))
            std_r = float(np.std(group_rewards))
            for rew, trs in zip(group_rewards, group_transitions_per_ep):
                adv = 0.0 if std_r < 1e-8 else (rew - mean_r) / (std_r + 1e-8)
                for tr in trs:
                    batch_transitions.append((tr, adv))
            update_rewards.extend(group_rewards)

        upd_stats = compute_update(policy, batch_transitions, optimizer)
        success_rate = float(np.mean(update_rewards))
        elapsed = time.time() - t0
        entry = {
            "update": update_idx, "success_rate": success_rate,
            "n_episodes": len(update_rewards), "elapsed_s": elapsed, **upd_stats,
        }
        progress.append(entry)
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log_json).write_text(json.dumps(progress, indent=2))
        print(f"===== UPDATE {update_idx}: success_rate={success_rate:.3f} "
              f"loss={upd_stats['loss']} grad_norm={upd_stats['grad_norm']} "
              f"elapsed={elapsed:.0f}s =====", flush=True)

    Path(args.out_ckpt).mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(args.out_ckpt)
    print(f"Saved RL checkpoint to {args.out_ckpt}")


if __name__ == "__main__":
    main()
