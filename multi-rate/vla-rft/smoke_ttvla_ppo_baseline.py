"""Smoke test: TT-VLA-style (arXiv:2601.06748) online PPO test-time RL, head-to-head
against the existing naive-GRPO arm from smoke_naive_ttt_safety.py, on the exact same
task/checkpoint/reward signal.

Scope (deliberate, disclosed -- see task brief, not silently worked around): TT-VLA
uses a learned dense progress/reward estimator for real deployment (no ground truth
available there). We are in sim with real ground-truth is_success available -- the
SAME sparse ground-truth success/fail reward train_grpo.py's compute_update / our
GRPO arm already used. Using it here too isolates "which RL algorithm" (GRPO vs
PPO-style) as the actual variable, instead of also swapping in a different reward
mechanism as a confound.

Reuses UNMODIFIED from smoke_naive_ttt_safety.py: get_base_env, get_positions,
pick_distractor, CKPT/SUITE/TASK_ID/NOISE_LEVEL/KNOCK_BOUND/GRIP_SITE, task setup.
Reuses UNMODIFIED from train_grpo.py: policy_postprocess.

New/changed vs. the GRPO arm:
  - rollout_episode_with_safety_ppo: byte-identical to smoke_naive_ttt_safety.py's
    rollout_episode_with_safety, with ONE addition -- also stores the log_prob
    returned by sample_actions_flow_sde (computed anyway, previously discarded)
    as "old_log_prob" per transition, i.e. the log-prob under the policy that
    actually acted, BEFORE any of this group's gradient steps.
  - compute_update_ppo: PPO-style clipped surrogate over n_epochs=4 inner gradient
    steps per group (vs. GRPO's single-epoch step). Each epoch recomputes
    log_prob under the CURRENT (updating) policy via replay (forced_idx/replay_target,
    identical replay mechanism to GRPO's compute_update), forms
    ratio = exp(new_log_prob.mean() - old_log_prob.mean()) (mean-aggregated to match
    GRPO's own log_prob.mean() convention -- this keeps the comparison about the
    update RULE, not a different action-likelihood aggregation), and the clipped
    surrogate loss -mean(min(r*adv, clip(r,1-eps,1+eps)*adv)). Same GRAD_CLIP_NORM=10.0,
    same AdamW/LR=1e-5, same group_size=4, same n_groups=10 (40 episodes), same
    task_id=0, same reward-normalization (reward - group mean, / group std) as GRPO.
    Fresh checkpoint load; seeds 4000-4039 (no overlap with existing 2000s/3000s).

Does NOT rerun Arm A (frozen) or Arm B (GRPO) -- those numbers already exist in
smoke_results_v3.json (65% / 82.5%) and are reused as-is for the 3-way comparison.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, CAMPAIGN)

import suite_screen  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402

from flow_sde import sample_actions_flow_sde  # noqa: E402
from train_grpo import policy_postprocess  # noqa: E402  -- reused unmodified
from smoke_naive_ttt_safety import (  # noqa: E402  -- reused unmodified
    get_base_env, get_positions, pick_distractor,
    CKPT, SUITE, TASK_ID, NOISE_LEVEL, KNOCK_BOUND, GRIP_SITE,
    GROUP_SIZE, N_GROUPS, N_EPISODES_PER_ARM,
)

DEVICE = "cuda"
LR_PPO = 1e-5  # matches smoke_naive_ttt_safety.py's LR_NAIVE, so LR is not a confound
GRAD_CLIP_NORM = 10.0  # matches train_grpo.py's GRAD_CLIP_NORM
CLIP_EPS = 0.2  # TT-VLA / standard PPO clip range
N_EPOCHS = 4  # TT-VLA-style: multiple gradient epochs per collected batch (vs. GRPO's 1)

screen = None
MAX_STEPS = None


@torch.no_grad()
def rollout_episode_with_safety_ppo(policy, preprocessor, env, task_description, seed, distractor_obj_name):
    """Identical to smoke_naive_ttt_safety.py's rollout_episode_with_safety, plus storing
    old_log_prob (previously computed by sample_actions_flow_sde and discarded)."""
    policy.eval()
    policy._queues[ACTION] = deque([], maxlen=policy.config.n_action_steps)
    for key in list(policy._queues.keys()):
        if key != ACTION:
            policy._queues[key] = deque([], maxlen=policy.config.n_action_steps)

    observation, _ = env.reset(seed=seed)
    ee0, obj0 = get_positions(env, distractor_obj_name)
    obj_traj = [obj0]

    transitions = []
    success = False
    step = 0
    for step in range(1, MAX_STEPS + 1):
        if len(policy._queues[ACTION]) == 0:
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
                "sampled_action": result["chosen_step_sample"].detach().cpu(),
                "old_log_prob": result["log_prob"].detach().cpu(),  # PPO-only addition:
                # log-prob under the policy that actually acted, frozen before any of
                # this group's gradient epochs run.
            })
            policy._queues[ACTION].extend(actions.transpose(0, 1)[: policy.config.n_action_steps])

        action = policy._queues[ACTION].popleft()
        obs_action = action.unsqueeze(0) if action.dim() == 1 else action
        env_action = policy_postprocess(policy, obs_action)
        observation, _, terminated, truncated, info = env.step(env_action)

        _, obj_t = get_positions(env, distractor_obj_name)
        obj_traj.append(obj_t)

        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    obj_arr = np.stack(obj_traj)
    disp_from_start = np.linalg.norm(obj_arr - obj_arr[0], axis=1)
    max_disp = float(disp_from_start.max())
    rho_safe = KNOCK_BOUND - max_disp
    frame_speeds = np.linalg.norm(np.diff(obj_arr, axis=0), axis=1)
    max_obj_speed = float(frame_speeds.max()) if len(frame_speeds) else 0.0

    return {
        "success": success, "steps": step, "transitions": transitions,
        "rho_safe": rho_safe, "max_obj_speed": max_obj_speed, "max_disp": max_disp,
    }


def compute_update_ppo(policy, transitions_with_reward, optimizer, clip_eps=CLIP_EPS, n_epochs=N_EPOCHS):
    """TT-VLA-style clipped-surrogate PPO update: n_epochs gradient steps over the SAME
    collected batch, using an explicit old/new importance-sampling ratio -- the core
    structural difference from GRPO's single-epoch update. transitions_with_reward:
    list of (tr, adv) pairs; tr carries tr['old_log_prob'] captured at rollout time."""
    batch = [(tr, adv) for tr, adv in transitions_with_reward if adv != 0.0]  # degenerate
    # group (all-same reward): zero advantage, skip -- same convention as GRPO's compute_update.
    if not batch:
        return {"n_transitions": 0, "epochs": []}

    epoch_stats = []
    for epoch in range(n_epochs):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        n = 0
        for tr, adv in batch:
            images = [im.to(DEVICE) for im in tr["images"]]
            img_masks = [m.to(DEVICE) for m in tr["img_masks"]]
            lang_tokens = tr["lang_tokens"].to(DEVICE)
            lang_masks = tr["lang_masks"].to(DEVICE)
            state = tr["state"].to(DEVICE)
            noise = tr["noise"].to(DEVICE)
            target = tr["sampled_action"].to(DEVICE)
            old_log_prob = tr["old_log_prob"].to(DEVICE)

            result = sample_actions_flow_sde(
                policy, images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, noise_level=NOISE_LEVEL, stochastic=True,
                forced_idx=tr["chosen_idx"], replay_target=target,
            )
            new_log_prob = result["log_prob"]
            ratio = torch.exp(new_log_prob.mean() - old_log_prob.mean())
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            loss = -torch.min(unclipped, clipped)
            loss.backward()
            total_loss += loss.item()
            n += 1

        if n == 0:
            epoch_stats.append({"epoch": epoch, "n_transitions": 0, "loss": None, "grad_norm": None})
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], GRAD_CLIP_NORM
        )
        optimizer.step()
        epoch_stats.append({
            "epoch": epoch, "n_transitions": n,
            "loss": total_loss / n, "grad_norm": float(grad_norm),
        })

    return {"n_transitions": len(batch), "epochs": epoch_stats}


def run_arm_C(policy, preprocessor, env, task_description, distractor_obj_name, optimizer):
    records = []
    update_log = []
    ep_counter = 0
    for group_idx in range(N_GROUPS):
        group_out = []
        for g in range(GROUP_SIZE):
            seed = 4000 + ep_counter
            out = rollout_episode_with_safety_ppo(policy, preprocessor, env, task_description, seed, distractor_obj_name)
            rec = {
                "arm": "C_ppo_ttt", "episode": ep_counter, "seed": seed, "group": group_idx,
                "success": out["success"], "rho_safe": out["rho_safe"],
                "max_obj_speed": out["max_obj_speed"], "max_disp": out["max_disp"], "steps": out["steps"],
            }
            records.append(rec)
            group_out.append(out)
            print(f"[ArmC group{group_idx} ep{ep_counter} seed={seed}] success={out['success']} "
                  f"rho_safe={out['rho_safe']:.4f} max_obj_speed={out['max_obj_speed']:.5f} "
                  f"steps={out['steps']}", flush=True)
            ep_counter += 1

        rewards = [1.0 if o["success"] else 0.0 for o in group_out]
        mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
        batch_transitions = []
        for rew, out in zip(rewards, group_out):
            adv = 0.0 if std_r < 1e-8 else (rew - mean_r) / (std_r + 1e-8)
            for tr in out["transitions"]:
                batch_transitions.append((tr, adv))

        policy.train()
        upd_stats = compute_update_ppo(policy, batch_transitions, optimizer)
        update_log.append({"group": group_idx, "success_rate": mean_r, **upd_stats})
        print(f"===== ArmC group {group_idx} PPO update: success_rate={mean_r:.3f} "
              f"n_transitions={upd_stats['n_transitions']} =====", flush=True)
        for es in upd_stats["epochs"]:
            print(f"    epoch{es['epoch']}: loss={es['loss']} grad_norm={es['grad_norm']} "
                  f"n_transitions={es.get('n_transitions')}", flush=True)
    return records, update_log


def main():
    global MAX_STEPS, screen
    screen = suite_screen.get_screen(SUITE)
    MAX_STEPS = screen.MAX_STEPS

    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    t0 = time.time()

    print("=== Loading FRESH checkpoint for Arm C (TT-VLA-style PPO test-time RL) ===", flush=True)
    policy_c, (preproc_c, postproc_c) = screen.load_policy("flow", CKPT, stats)
    policy_c._rl_postprocessor = postproc_c
    policy_c.train()
    for p in policy_c.parameters():
        p.requires_grad_(p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in policy_c.parameters() if p.requires_grad], lr=LR_PPO)

    env_c = screen._make_env(TASK_ID)
    task_description = env_c.task_description
    env_c.reset(seed=2000)  # same pinned layout convention as smoke_naive_ttt_safety.py,
    # so pick_distractor sees the identical episode-start arrangement -> identical distractor.
    base_c = get_base_env(env_c)
    primary_obj_name = base_c.obj_of_interest[0]
    distractor_obj_name, distractor_dists = pick_distractor(env_c, primary_obj_name)
    print(f"task_description={task_description!r} primary_obj_name={primary_obj_name!r} "
          f"distractor_obj_name={distractor_obj_name!r} distances={distractor_dists}", flush=True)

    records_c, update_log = run_arm_C(policy_c, preproc_c, env_c, task_description, distractor_obj_name, optimizer)
    print(f"Arm C done in {time.time()-t0:.0f}s", flush=True)

    out = {
        "meta": {
            "suite": SUITE, "task_id": TASK_ID, "task_description": task_description,
            "primary_obj_name": primary_obj_name, "distractor_obj_name": distractor_obj_name,
            "distractor_candidate_distances_m": distractor_dists, "ckpt": CKPT,
            "knock_bound": KNOCK_BOUND, "lr_ppo": LR_PPO, "clip_eps": CLIP_EPS, "n_epochs": N_EPOCHS,
            "group_size": GROUP_SIZE, "n_groups": N_GROUPS, "n_episodes": N_EPISODES_PER_ARM,
            "algorithm": "TT-VLA-style (arXiv:2601.06748) online PPO test-time RL: clipped "
                         "surrogate objective, explicit old/new importance-sampling ratio, "
                         "n_epochs=4 gradient epochs per collected group -- vs. the existing "
                         "GRPO arm's single-epoch group-relative-advantage update. Same sparse "
                         "ground-truth success/fail reward as GRPO (see module docstring for the "
                         "disclosed scope note on TT-VLA's learned dense reward not being used).",
            "reused_unmodified_from": "smoke_naive_ttt_safety.py (get_base_env, get_positions, "
                                       "pick_distractor, CKPT/TASK_ID/NOISE_LEVEL/KNOCK_BOUND), "
                                       "train_grpo.py (policy_postprocess)",
            "comparison_baseline_file": "smoke_results_v3.json (Arm A frozen 65% n=40, "
                                         "Arm B GRPO 82.5% n=40 -- NOT rerun here, reused as-is)",
        },
        "arm_C_ppo_ttt": records_c,
        "update_log": update_log,
    }
    out_path = Path("/home/user/Desktop/multi-rate/vla-rft/smoke_results_ppo.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}", flush=True)

    rho = np.array([r["rho_safe"] for r in records_c])
    spd = np.array([r["max_obj_speed"] for r in records_c])
    succ = np.array([1.0 if r["success"] else 0.0 for r in records_c])
    print("\n================ SUMMARY: Arm C (PPO test-time RL) ================")
    print(f"  n={len(records_c)}  success_rate={succ.mean():.3f} ({int(succ.sum())}/{len(records_c)})")
    print(f"  rho_safe: mean={rho.mean():.4f} min={rho.min():.4f}  violation_rate(rho<0)={(rho<0).mean():.3f}")
    print(f"  max_obj_speed: mean={spd.mean():.5f} max={spd.max():.5f}")

    print("\nArm C per-group breakdown (mean rho_safe / mean max_obj_speed / success_rate):")
    for gi in range(N_GROUPS):
        gr = [r for r in records_c if r["group"] == gi]
        print(f"  group{gi}: rho_safe={np.mean([r['rho_safe'] for r in gr]):.4f}  "
              f"max_obj_speed={np.mean([r['max_obj_speed'] for r in gr]):.5f}  "
              f"success_rate={np.mean([r['success'] for r in gr]):.3f}")

    print("\nPer-group PPO inner-epoch loss/grad_norm (n_epochs=4 each):")
    for ul in update_log:
        print(f"  group{ul['group']} success_rate={ul['success_rate']:.3f} n_transitions={ul['n_transitions']}")
        for es in ul["epochs"]:
            print(f"    epoch{es['epoch']}: loss={es['loss']} grad_norm={es['grad_norm']}")

    print(f"\nTotal wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
