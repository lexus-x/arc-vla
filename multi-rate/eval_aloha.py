#!/usr/bin/env python3
"""
Evaluates LeRobot ACT policy on Aloha Peg Insertion under Action Rate Decimation (k=2, 4).
Compares TAC-Fold + satfix against standard baselines:
  - native (1X 50Hz reference)
  - zoh (zero-order hold)
  - spline (unconstrained cubic spline)
  - bspline (unconstrained B-spline with curvature regularization)
  - tac_fold_satfix (operator folding + saturation projection)

Usage:
  MUJOCO_GL=egl python eval_aloha.py --k 2 --n_episodes 50
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FULL_GRID_DIR = HERE / "full_grid_2026-09-07"
sys.path.insert(0, str(FULL_GRID_DIR))

# Import resamplers from validated multi-rate library
from resample_math import (
    coarsen_delta,
    resample_zoh,
    resample_tac_fold_satfix,
    resample_spline,
    resample_bspline,
    RESAMPLERS,
)

# Ensure bspline eps is registered
from resample_bspline2 import resample_bspline_eps
RESAMPLERS["bspline"] = lambda b, k: resample_bspline_eps(b, k, eps=0.005)

import gymnasium as gym
import gym_aloha
from safetensors.torch import load_file
from lerobot.policies.act.modeling_act import ACTPolicy


def exact_mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    b_only = int((a & ~b).sum())  # a success, b fail
    c_only = int((~a & b).sum())  # a fail, b success
    d = b_only + c_only
    if d == 0:
        return b_only, c_only, 1.0
    p = float(min(1.0, 2.0 * sum(math.comb(d, i) for i in range(min(b_only, c_only) + 1)) / (2**d)))
    return b_only, c_only, p


def resample_chunk(chunk_np: np.ndarray, k: int, arm: str) -> np.ndarray:
    """
    Decimate action chunk (T, 14) by factor k, then resample back using specified arm.
    """
    if arm == "native":
        return chunk_np.copy()
    
    T, D = chunk_np.shape
    n_blocks = T // k
    remainder = chunk_np[n_blocks * k:]
    if n_blocks == 0:
        return chunk_np.copy()
    
    block_sum = coarsen_delta(chunk_np, k)
    fn = RESAMPLERS[arm]
    reconstructed = fn(block_sum, k)
    if len(remainder) > 0:
        reconstructed = np.concatenate([reconstructed, remainder], axis=0)
    return reconstructed.astype(np.float32)


def run_episode(env, policy, norm_stats, seed, k, arm, max_steps=400, device="cuda"):
    """
    Run one evaluation episode with the given arm under seed using exact safetensors normalization.
    """
    policy.eval()
    policy.reset()
    
    obs, info = env.reset(seed=seed)
    
    state_mean = norm_stats["state_mean"]
    state_std = norm_stats["state_std"]
    img_mean = norm_stats["img_mean"]
    img_std = norm_stats["img_std"]
    act_mean = norm_stats["act_mean"]
    act_std = norm_stats["act_std"]
    
    step = 0
    is_success = False
    max_reward = 0.0
    
    while step < max_steps:
        # Preprocess top camera image
        top_img = torch.from_numpy(obs["top"]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        top_img = (top_img - img_mean) / img_std
        
        # Preprocess joint state (14-dim)
        state_raw = torch.from_numpy(env.unwrapped._env.physics.data.qpos[:14].copy()).float().unsqueeze(0).to(device)
        state = (state_raw - state_mean) / state_std
        
        batch = {
            "observation.images.top": top_img,
            "observation.state": state,
        }
        
        if len(policy._action_queue) == 0:
            with torch.inference_mode():
                # Policy outputs normalized actions (1, 100, 14)
                raw_actions = policy.predict_action_chunk(batch)[0] # (100, 14)
                
                # Unnormalize to real actuator domain
                real_actions = raw_actions * act_std + act_mean
                chunk_np = real_actions.cpu().numpy()
                
                if arm == "native":
                    policy._action_queue.extend(real_actions)
                else:
                    # Resample chunk in real action space
                    resampled_np = resample_chunk(chunk_np, k, arm)
                    resampled_t = torch.from_numpy(resampled_np).to(device, raw_actions.dtype)
                    policy._action_queue.extend(resampled_t)
        
        act = policy._action_queue.popleft().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(act)
        
        if reward > max_reward:
            max_reward = float(reward)
        if reward >= 4.0:
            is_success = True
            break
            
        if terminated or truncated:
            break
        step += 1
        
    return is_success, max_reward, step


def main():
    parser = argparse.ArgumentParser(description="Aloha Peg Insertion Multi-Rate Resampler Evaluation")
    parser.add_argument("--k", type=int, default=2, choices=[2, 4], help="Decimation factor")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of paired test episodes")
    parser.add_argument("--arms", type=str, default="native,zoh,spline,bspline,tac_fold_satfix", help="Comma-separated list of arms")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--seed_start", type=int, default=0, help="Starting seed")
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--out_dir", type=str, default="multi-rate/aloha_eval_results", help="Output directory")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_aloha_insertion_k{args.k}.json"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Aloha Peg Insertion Evaluation (k={args.k}, n={args.n_episodes}) on {device} ===")
    print(f"Arms to evaluate: {arms}")

    # Environment
    env = gym.make("gym_aloha/AlohaInsertion-v0")

    # Pretrained policy & normalization stats
    repo_id = "lerobot/act_aloha_sim_insertion_human"
    print(f"Loading pretrained ACT policy: {repo_id} ...")
    policy = ACTPolicy.from_pretrained(repo_id)
    policy.eval()
    policy.to(device)

    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    tensors = load_file(weights_path)
    
    norm_stats = {
        "state_mean": tensors["normalize_inputs.buffer_observation_state.mean"].to(device),
        "state_std": tensors["normalize_inputs.buffer_observation_state.std"].to(device),
        "img_mean": tensors["normalize_inputs.buffer_observation_images_top.mean"].to(device),
        "img_std": tensors["normalize_inputs.buffer_observation_images_top.std"].to(device),
        "act_mean": tensors["unnormalize_outputs.buffer_action.mean"].to(device),
        "act_std": tensors["unnormalize_outputs.buffer_action.std"].to(device),
    }

    results = {
        "k": args.k,
        "n_episodes": args.n_episodes,
        "task": "AlohaInsertion-v0",
        "model": repo_id,
        "arms": arms,
        "success": {arm: [] for arm in arms},
        "max_rewards": {arm: [] for arm in arms},
        "episode_lengths": {arm: [] for arm in arms},
        "success_rate": {},
        "mcnemar_vs_tacfold": {},
    }

    t0 = time.time()
    for ep in range(args.n_episodes):
        seed = args.seed_start + ep
        ep_summary = []
        for arm in arms:
            succ, max_r, length = run_episode(
                env, policy, norm_stats,
                seed=seed, k=args.k, arm=arm, max_steps=args.max_steps, device=device
            )
            results["success"][arm].append(succ)
            results["max_rewards"][arm].append(max_r)
            results["episode_lengths"][arm].append(length)
            ep_summary.append(f"{arm}: {'SUCC' if succ else 'FAIL'}(r={max_r:.0f})")

        elapsed = time.time() - t0
        print(f"Episode {ep+1:2d}/{args.n_episodes} (seed {seed}) [{elapsed/60:.1f}m]: | " + " | ".join(ep_summary))

    env.close()

    print("\n=== FINAL RESULTS SUMMARY ===")
    for arm in arms:
        sr = float(np.mean(results["success"][arm]))
        results["success_rate"][arm] = sr
        print(f"  {arm:18s}: {sum(results['success'][arm])}/{args.n_episodes} ({sr*100:5.1f}%)")

    # Paired McNemar against TAC-Fold + satfix
    if "tac_fold_satfix" in results["success"]:
        tf_succ = results["success"]["tac_fold_satfix"]
        print("\n=== PAIRED EXACT McNEMAR STATS (TAC-Fold+satfix vs. Baselines) ===")
        for arm in arms:
            if arm == "tac_fold_satfix":
                continue
            ref_succ = results["success"][arm]
            b, c, p = exact_mcnemar(tf_succ, ref_succ)
            delta = float(np.mean(tf_succ) - np.mean(ref_succ)) * 100.0
            results["mcnemar_vs_tacfold"][arm] = {
                "delta_pp": delta,
                "tac_only": b,
                "ref_only": c,
                "p_value_raw": p,
            }
            print(f"  vs {arm:16s}: {delta:+5.1f} pp (discordant {b}/{c}, raw p = {p:.4g})")

    results["total_time_sec"] = time.time() - t0
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults successfully written to: {out_file}")


if __name__ == "__main__":
    main()
