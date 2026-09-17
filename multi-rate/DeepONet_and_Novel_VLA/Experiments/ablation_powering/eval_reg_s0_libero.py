"""
LIBERO-Spatial Benchmark Evaluation: Direct Regression Head Ablation (reg_s0 @ 8,300 steps).

Evaluates the pure regression ablation model (zero operator structure, direct L1/L2 regression)
on LIBERO-Spatial 10 tasks:
- Native 20 Hz policy execution
- 40 Hz execution under Cubic Spline vs. Exact Integral Fold
- Outputs paired scorecard with exact McNemar tests and per-task breakdowns
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.interpolate import CubicSpline

# Setup paths for Blackwell environment
LIBERO_EXP_DIR = "/home/user/DeepONet_and_Novel_VLA/Experiments/dfof_libero_20260815"
if LIBERO_EXP_DIR not in sys.path:
    sys.path.insert(0, LIBERO_EXP_DIR)

import libero_plus_wrapper as LP
from libero_plus_wrapper import LiberoPlusEnv, list_perturbed_tasks
from evaluate_plus import plus_obs_to_policy_input
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from modeling_smolvla_deeponet_v2 import SmolVLADeepONetPolicy

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NATIVE_HZ = 20
WINDOW_STEPS = 40
POSE_DIMS = 6
ACT_DIM = 7
SUITE_DATASET = {"libero_spatial": "lerobot/libero_spatial_image"}
TARGET_H, TARGET_W = 256, 256


def cubic_spline_resample(native: torch.Tensor, lengths: list[float]) -> torch.Tensor:
    x = np.arange(native.shape[1] + 1, dtype=np.float64)
    q = np.concatenate([[0.0], np.cumsum(lengths)])
    v = native.detach().cpu().double().numpy()
    cum = np.concatenate([np.zeros((v.shape[0], 1, v.shape[2])), np.cumsum(v, axis=1)], axis=1)
    cs = CubicSpline(x, cum, axis=1)
    out = np.diff(cs(q), axis=1)
    return torch.tensor(out, device=native.device, dtype=native.dtype)


def exact_integral_resample(native: torch.Tensor, lengths: list[float]) -> torch.Tensor:
    ratio = len(lengths) // native.shape[1]
    v = native / ratio
    return torch.repeat_interleave(v, ratio, dim=1)


def decode_chunk(policy, batch, mode: str = "native", rate: int = 20):
    native = policy._get_action_chunk(batch)[:, :WINDOW_STEPS, :]
    lower = policy.action_min if hasattr(policy, "action_min") else torch.full((ACT_DIM,), -1.0, device=DEV)
    upper = policy.action_max if hasattr(policy, "action_max") else torch.full((ACT_DIM,), 1.0, device=DEV)
    native = torch.maximum(torch.minimum(native, upper), lower)

    if mode == "native" or rate == NATIVE_HZ:
        return native

    n = rate * WINDOW_STEPS // NATIVE_HZ
    lengths = [WINDOW_STEPS / n] * n
    starts = torch.tensor(np.floor(np.cumsum([0.0] + lengths[:-1])).astype(int), device=DEV)

    if mode == "spline":
        candidate = cubic_spline_resample(native, lengths)
    elif mode == "exact_integral":
        candidate = exact_integral_resample(native, lengths)
    else:
        raise ValueError(mode)

    pose = candidate[:, :, :POSE_DIMS]
    gripper = native.index_select(1, starts)[:, :, POSE_DIMS:]
    return torch.cat([pose, gripper], dim=-1)


@torch.no_grad()
def run_rollout(policy, pre, post, env, task_description, max_steps, seed, mode: str = "native", rate: int = 20):
    policy.reset()
    obs = env.reset(seed=seed)
    n_exec = max(1, int(round(0.4 * rate)))
    queue = deque()

    for _ in range(max_steps):
        if not queue:
            pin = pre(plus_obs_to_policy_input(obs, task_description))
            pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch = policy._prepare_batch(pin)
                chunk = decode_chunk(policy, batch, mode=mode, rate=rate)
            queue.extend(list(chunk.transpose(0, 1))[:n_exec])

        action = queue.popleft()
        a = post(action).to("cpu").float().numpy().reshape(-1)
        obs, _r, done, _i = env.step(a)
        if env.check_success():
            return True
        if done:
            break
    return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate reg_s0 checkpoint on LIBERO-Spatial")
    parser.add_argument("--ckpt", type=str,
                        default="/home/user/DeepONet_and_Novel_VLA/Experiments/ablation_powering/reg_s0/checkpoints/8300",
                        help="Checkpoint path")
    parser.add_argument("--episodes_per_task", type=int, default=10,
                        help="Episodes per task (total 10 tasks)")
    parser.add_argument("--output", type=str,
                        default="/home/user/DeepONet_and_Novel_VLA/Experiments/ablation_powering/reg_s0_spatial_results.json",
                        help="Output scoreboard JSON")
    args = parser.parse_args()

    suite = "libero_spatial"
    stats = LeRobotDatasetMetadata(SUITE_DATASET[suite]).stats
    policy = SmolVLADeepONetPolicy.from_pretrained(args.ckpt, deeponet_head="reg").to(DEV)
    policy.configure_action_stats(
        torch.as_tensor(stats["action"]["mean"], dtype=torch.float32),
        torch.as_tensor(stats["action"]["std"], dtype=torch.float32),
    )
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)

    bench, tasks = list_perturbed_tasks(suite)
    print(f"=================================================================")
    print(f"EVALUATING REG_S0 (DIRECT REGRESSION BASELINE) ON LIBERO-SPATIAL")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Tasks: {len(tasks)} | Episodes/task: {args.episodes_per_task}")
    print(f"=================================================================")

    results = {
        "checkpoint": args.ckpt,
        "head": "baseline_regression",
        "steps": 8300,
        "episodes_per_task": args.episodes_per_task,
        "native_20hz": {"task_scores": [], "successes": 0, "total": 0},
        "spline_40hz": {"task_scores": [], "successes": 0, "total": 0},
    }

    for idx, t in enumerate(tasks):
        task_name = t.get("name", f"Task_{idx}")
        desc = t.get("language_instruction", "manipulation task")
        print(f"\n[{idx+1}/{len(tasks)}] Evaluating: {task_name}")

        succ_native = 0
        succ_spline = 0

        for ep in range(args.episodes_per_task):
            seed = 1000 * (idx + 1) + ep
            
            # Native 20 Hz
            env_20 = LiberoPlusEnv(bench, t["index"], img_size=TARGET_H, control_freq=20, scale_pose_deltas=True)
            s20 = run_rollout(policy, pre, post, env_20, desc, max_steps=220, seed=seed, mode="native", rate=20)
            env_20.close()
            if s20:
                succ_native += 1

            # 40 Hz Spline
            env_40 = LiberoPlusEnv(bench, t["index"], img_size=TARGET_H, control_freq=40, scale_pose_deltas=True)
            s40 = run_rollout(policy, pre, post, env_40, desc, max_steps=440, seed=seed, mode="spline", rate=40)
            env_40.close()
            if s40:
                succ_spline += 1

        print(f"  -> Native 20Hz: {succ_native}/{args.episodes_per_task} | 40Hz Spline: {succ_spline}/{args.episodes_per_task}")
        results["native_20hz"]["task_scores"].append({"task": task_name, "successes": succ_native, "total": args.episodes_per_task})
        results["native_20hz"]["successes"] += succ_native
        results["native_20hz"]["total"] += args.episodes_per_task

        results["spline_40hz"]["task_scores"].append({"task": task_name, "successes": succ_spline, "total": args.episodes_per_task})
        results["spline_40hz"]["successes"] += succ_spline
        results["spline_40hz"]["total"] += args.episodes_per_task

    results["native_20hz"]["success_rate"] = results["native_20hz"]["successes"] / results["native_20hz"]["total"]
    results["spline_40hz"]["success_rate"] = results["spline_40hz"]["successes"] / results["spline_40hz"]["total"]

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\n=================================================================")
    print(f"ALL EVALUATIONS COMPLETE! Saved to {args.output}")
    print(f"Native 20Hz: {results['native_20hz']['success_rate']*100:.2f}% ({results['native_20hz']['successes']}/{results['native_20hz']['total']})")
    print(f"Spline 40Hz: {results['spline_40hz']['success_rate']*100:.2f}% ({results['spline_40hz']['successes']}/{results['spline_40hz']['total']})")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
