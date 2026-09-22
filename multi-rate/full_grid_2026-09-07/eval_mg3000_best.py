#!/usr/bin/env python3
"""Evaluate RoboCasa 3,000-demo Diffusion Policy Best Epoch Checkpoints.

Runs closed-loop rollouts for the best checkpoints of:
- CloseSingleDoor: model_epoch_990.pth
- CoffeePressButton: model_epoch_580.pth
- TurnOffMicrowave: model_epoch_970.pth
- TurnOffSinkFaucet: model_epoch_970.pth
"""
import argparse
import json
import os
import sys
import time
import numpy as np

# Ensure robomimic and robocasa are available in sys.path
ROBOMIMIC_DIR = "/tmp/claude-1000/-home-user-Desktop/eca0093a-6f27-4f08-b55b-27d2b8c90be8/scratchpad/robomimic"
ROBOCASA_DIR = "/home/user/Isaac-GR00T/external_dependencies/robocasa"

if ROBOMIMIC_DIR not in sys.path:
    sys.path.insert(0, ROBOMIMIC_DIR)
if ROBOCASA_DIR not in sys.path:
    sys.path.insert(0, ROBOCASA_DIR)

import torch
import robosuite
import robocasa
import robomimic
import robomimic.utils.file_utils as FU
import robomimic.utils.env_utils as EU

BEST_CHECKPOINTS = {
    "CloseSingleDoor": {
        "epoch": 990,
        "path": "/media/user/C2FE578FFE577A9D/robocasa_mg3000_ckpts/seed_123_ds_CloseSingleDoor/20260917151903/models/model_epoch_990.pth",
    },
    "CoffeePressButton": {
        "epoch": 580,
        "path": "/media/user/C2FE578FFE577A9D/robocasa_mg3000_ckpts/seed_123_ds_CoffeePressButton/20260917151903/models/model_epoch_580.pth",
    },
    "TurnOffMicrowave": {
        "epoch": 970,
        "path": "/media/user/C2FE578FFE577A9D/robocasa_mg3000_ckpts/seed_123_ds_TurnOffMicrowave/20260917151903/models/model_epoch_970.pth",
    },
    "TurnOffSinkFaucet": {
        "epoch": 970,
        "path": "/media/user/C2FE578FFE577A9D/robocasa_mg3000_ckpts/seed_123_ds_TurnOffSinkFaucet/20260917151903/models/model_epoch_970.pth",
    },
}


def evaluate_task(task_name, ckpt_path, epoch, n_rollouts=25, horizon=500, seed=0, output_dir=None):
    print(f"\n========================================================")
    print(f"Starting Evaluation: {task_name} (Epoch {epoch})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Rollouts: {n_rollouts}, Max Horizon: {horizon}, Seed: {seed}")
    print(f"========================================================")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return None

    # Load policy and config
    policy, ckpt_dict = FU.policy_from_checkpoint(ckpt_path=ckpt_path, device="cpu", verbose=False)
    config, _ = FU.config_from_checkpoint(ckpt_dict=ckpt_dict)

    # Create and wrap environment
    env = EU.create_env_from_metadata(
        env_meta=ckpt_dict["env_metadata"],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    env = EU.wrap_env_from_config(env, config=config)

    successes = []
    episode_lengths = []
    t_start = time.time()

    for ep_idx in range(n_rollouts):
        ep_seed = seed * 1000 + ep_idx
        if hasattr(env, "set_seed"):
            env.set_seed(ep_seed)
        elif hasattr(env.env, "set_seed"):
            env.env.set_seed(ep_seed)

        policy.start_episode()
        policy._ep_lang_emb = np.zeros((1, 768), dtype=np.float32)

        obs = env.reset()
        if "lang_emb" not in obs:
            obs["lang_emb"] = np.zeros((config.train.frame_stack, 768), dtype=np.float32)

        ep_success = False
        steps = 0
        ep_t0 = time.time()

        for step in range(horizon):
            if "lang_emb" not in obs:
                obs["lang_emb"] = np.zeros((config.train.frame_stack, 768), dtype=np.float32)

            act = policy(obs)
            obs, reward, done, info = env.step(act)
            steps += 1

            check_succ = env.is_success()
            if isinstance(check_succ, dict):
                ep_success = bool(check_succ.get("task", False))
            else:
                ep_success = bool(check_succ)

            if ep_success or done:
                break

        ep_duration = time.time() - ep_t0
        successes.append(ep_success)
        episode_lengths.append(steps)

        curr_sr = 100.0 * np.mean(successes)
        print(f"[{task_name}] Ep {ep_idx+1:02d}/{n_rollouts:02d} | "
              f"Success: {ep_success} | Steps: {steps:3d} | "
              f"Time: {ep_duration:.1f}s | Current SR: {curr_sr:.1f}% ({sum(successes)}/{len(successes)})",
              flush=True)

    total_time = time.time() - t_start
    final_sr = float(np.mean(successes))
    result = {
        "task": task_name,
        "epoch": epoch,
        "checkpoint": ckpt_path,
        "n_rollouts": n_rollouts,
        "horizon": horizon,
        "seed": seed,
        "success_rate": final_sr,
        "success_count": int(sum(successes)),
        "episode_lengths": episode_lengths,
        "success_list": [bool(s) for s in successes],
        "total_duration_sec": total_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"\n---> {task_name} Finished! Success Rate: {final_sr * 100:.1f}% "
          f"({sum(successes)}/{n_rollouts}) in {total_time:.1f}s")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_json = os.path.join(output_dir, f"eval_mg3000_{task_name}_epoch_{epoch}.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to: {out_json}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate RoboCasa mg3000 best checkpoints")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "CloseSingleDoor", "CoffeePressButton", "TurnOffMicrowave", "TurnOffSinkFaucet"])
    parser.add_argument("--n_rollouts", type=int, default=20, help="Number of rollouts per task")
    parser.add_argument("--horizon", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="/home/user/Desktop/multi-rate/full_grid_2026-09-07/mg3000_eval_results")
    args = parser.parse_args()

    tasks_to_run = list(BEST_CHECKPOINTS.keys()) if args.task == "all" else [args.task]

    overall_results = {}
    for task_name in tasks_to_run:
        info = BEST_CHECKPOINTS[task_name]
        res = evaluate_task(
            task_name=task_name,
            ckpt_path=info["path"],
            epoch=info["epoch"],
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        if res:
            overall_results[task_name] = res

    # Summary table
    print("\n" + "="*70)
    print(f"{'Task':<22} | {'Epoch':<6} | {'Success Rate':<14} | {'Wins / Total':<12}")
    print("-" * 70)
    for t, r in overall_results.items():
        sr_pct = r['success_rate'] * 100.0
        wins = r['success_count']
        tot = r['n_rollouts']
        print(f"{t:<22} | {r['epoch']:<6} | {sr_pct:5.1f}%        | {wins}/{tot}")
    print("="*70)


if __name__ == "__main__":
    main()
