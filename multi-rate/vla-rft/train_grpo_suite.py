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
import itertools
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, CAMPAIGN)

import evaluate_height_screen as screen  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402

from flow_sde import sample_actions_flow_sde  # noqa: E402

DEVICE = "cuda"
NOISE_LEVEL = 0.1

DATASET_BY_SUITE = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_10": "lerobot/libero_10_image",
}
# Vault's own documented per-suite horizon convention (evaluate_multirate_honest.py
# SUITE_WALLCLOCK comment): standard LIBERO max_steps @20Hz native control freq.
# Reusing Spatial's 220 for other suites silently truncates them (that module's own
# warning) -- NOT a re-tune of RL hyperparameters, this is the same physical-time
# convention already used for eval, applied consistently to training rollouts too.
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
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


def compute_update(policy, transitions_with_advantage, optimizer, entropy_coef=0.0):
    """One gradient step over a batch of (transition, advantage) pairs, GRPO-style:
    loss = -mean_i[ advantage_i * log_prob_i ], log_prob_i recomputed under current
    params via replay_target (see flow_sde.sample_actions_flow_sde docstring).

    entropy_coef > 0 adds +entropy_coef * log_prob.mean() to the loss, i.e. an
    exploration bonus that pushes the sampled point's log-density down (more
    spread out / less collapsed), applied on the same non-degenerate transitions
    the advantage term uses. Off (0.0) reproduces the original PHASE3/PHASE4
    vanilla-GRPO behavior exactly."""
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
        if entropy_coef:
            loss = loss + entropy_coef * log_prob.mean()
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
    parser.add_argument("--suite", required=True, choices=list(DATASET_BY_SUITE))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0,
                         help="RNG seed: offsets rollout env-reset seeds into a "
                              "disjoint block per seed, and seeds task-cycling/"
                              "resample RNG + torch/numpy, so --seed 0/1/2 are "
                              "genuinely different runs, not the same rollouts.")
    parser.add_argument("--full_task_coverage", action="store_true",
                         help="Cycle through all 10 suite tasks across updates "
                              "instead of the pinned 2-task (3,5) pilot scope.")
    parser.add_argument("--dynamic_sampling", action="store_true",
                         help="DAPO-style: when a group is degenerate (zero reward "
                              "variance), resample it with fresh rollouts (up to "
                              "--max_resample times) instead of discarding the "
                              "gradient signal, to counter GRPO advantage collapse.")
    parser.add_argument("--max_resample", type=int, default=2,
                         help="Resample budget per degenerate group (only used "
                              "with --dynamic_sampling).")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                         help="Exploration-bonus coefficient (0.0 = off, matches "
                              "original vanilla-GRPO pilot exactly).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    # Monkey-patch evaluate_height_screen's module-level suite constants, same
    # pattern evaluate_multirate_honest.py itself uses for cross-suite eval
    # ("screen.SUITE = suite").
    screen.SUITE = args.suite
    screen.DATASET = DATASET_BY_SUITE[args.suite]
    screen.MAX_STEPS = MAX_STEPS_BY_SUITE[args.suite]
    CKPT_IN = args.ckpt

    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preprocessor, postprocessor) = screen.load_policy("flow", CKPT_IN, stats)
    policy._rl_postprocessor = postprocessor
    policy.train()
    for p in policy.parameters():
        p.requires_grad_(p.requires_grad)  # keep checkpoint's own frozen/unfrozen split

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=LR
    )

    # Task pool: pinned (3, 5) pilot scope, or all 10 standard LIBERO task ids
    # (valid 0-9 in every suite) cycled --n_groups_per_update at a time so every
    # task gets trained on over the course of a run, not just 2 of 10.
    task_pool = list(range(10)) if args.full_task_coverage else list(screen.TASK_IDS)
    rng.shuffle(task_pool)
    task_cycle = itertools.cycle(task_pool)
    task_envs = {tid: screen._make_env(tid) for tid in task_pool}

    progress = []
    ep_counter = 0
    seed_block = args.seed * 1_000_000  # disjoint rollout-seed range per --seed
    t0 = time.time()
    for update_idx in range(args.n_updates):
        batch_transitions = []
        update_rewards = []
        update_tids = [next(task_cycle) for _ in range(args.n_groups_per_update)]
        for tid in update_tids:
            env = task_envs[tid]
            group_id = f"u{update_idx}_t{tid}"

            max_attempts = 1 + (args.max_resample if args.dynamic_sampling else 0)
            for attempt in range(max_attempts):
                group_rewards = []
                group_transitions_per_ep = []
                for g in range(args.group_size):
                    seed = 1000 + seed_block + ep_counter
                    ep_counter += 1
                    env.init_state_id = g % max(1, screen.TRIALS_PER_TASK)
                    out = rollout_episode(policy, preprocessor, env, env.task_description, seed=seed, group_id=group_id)
                    reward = 1.0 if out["success"] else 0.0
                    group_rewards.append(reward)
                    group_transitions_per_ep.append(out["transitions"])
                    print(f"[update {update_idx}] task={tid} attempt={attempt} rollout={g} "
                          f"success={out['success']} steps={out['steps']} "
                          f"n_chunks={len(out['transitions'])}", flush=True)
                std_r = float(np.std(group_rewards))
                if std_r >= 1e-8 or attempt == max_attempts - 1:
                    break  # non-degenerate, or resample budget exhausted -- accept as-is
                print(f"[update {update_idx}] task={tid} degenerate group "
                      f"(all reward={group_rewards[0]}), resampling "
                      f"({attempt + 1}/{args.max_resample})", flush=True)

            mean_r = float(np.mean(group_rewards))
            for rew, trs in zip(group_rewards, group_transitions_per_ep):
                adv = 0.0 if std_r < 1e-8 else (rew - mean_r) / (std_r + 1e-8)
                for tr in trs:
                    batch_transitions.append((tr, adv))
            update_rewards.extend(group_rewards)

        upd_stats = compute_update(policy, batch_transitions, optimizer, entropy_coef=args.entropy_coef)
        success_rate = float(np.mean(update_rewards))
        elapsed = time.time() - t0
        entry = {
            "update": update_idx, "success_rate": success_rate,
            "n_episodes": len(update_rewards), "elapsed_s": elapsed,
            "task_ids": update_tids, **upd_stats,
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
