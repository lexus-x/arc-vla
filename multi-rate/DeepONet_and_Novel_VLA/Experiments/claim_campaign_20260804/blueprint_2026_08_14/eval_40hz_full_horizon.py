"""Full-Horizon 40 Hz Evaluation across all 10 ManiSkill Tasks.

Sets max_episode_steps=300 and catches task-internal buffer bounds (e.g. DrawTriangle dot buffers).
"""

from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator
import mani_skill.envs

OUT_DIR = os.path.expanduser("~/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14")
VIDEO_DIR = os.path.join(OUT_DIR, "comparison_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

TASKS_CONFIG = [
    {
        "name": "LiftPegUpright-v1",
        "h5": "/home/user/maniskill_data/raw/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PickCube-v1",
        "h5": "/home/user/maniskill_data/pick_rl_joint.h5",
        "json": "/home/user/maniskill_data/pick_rl_joint.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "StackCube-v1",
        "h5": "/home/user/maniskill_data/raw/StackCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/StackCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PokeCube-v1",
        "h5": "/home/user/maniskill_data/raw/PokeCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/PokeCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PullCube-v1",
        "h5": "/home/user/maniskill_data/raw/PullCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/PullCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PushCube-v1",
        "h5": "/home/user/maniskill_data/raw/PushCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/PushCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "RollBall-v1",
        "h5": "/home/user/maniskill_data/raw/RollBall-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/RollBall-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PegInsertionSide-v1",
        "h5": "/home/user/maniskill_data/raw/PegInsertionSide-v1/rl/trajectory.h5",
        "json": "/home/user/maniskill_data/raw/PegInsertionSide-v1/rl/trajectory.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PushT-v1",
        "h5": "/home/user/maniskill_data/pusht_rl.h5",
        "json": "/home/user/maniskill_data/pusht_rl.json",
        "control_mode": "pd_ee_delta_pose",
        "has_gripper": True,
    },
    {
        "name": "DrawTriangle-v1",
        "h5": "/home/user/maniskill_data/raw/DrawTriangle-v1/motionplanning/trajectory.h5",
        "json": "/home/user/maniskill_data/raw/DrawTriangle-v1/motionplanning/trajectory.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
]

TARGET_H, TARGET_W = 512, 512


def typed_coarse(actions: np.ndarray, has_gripper: bool) -> tuple[np.ndarray, np.ndarray | None]:
    paired = actions.reshape(len(actions) // 2, 2, actions.shape[1])
    if has_gripper:
        delta = paired[:, :, :-1].sum(axis=1)
        gripper = paired[:, 0, -1:]
        return delta, gripper
    else:
        delta = paired.sum(axis=1)
        return delta, None


def pchip_fold_resample(delta: np.ndarray, gripper: np.ndarray | None, target_len: int) -> np.ndarray:
    source_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)],
        axis=0,
    )
    target_t = np.linspace(0.0, float(source_len), target_len + 1)
    target_path = PchipInterpolator(source_t, cumulative, axis=0)(target_t)
    pose = np.diff(target_path, axis=0)

    if gripper is not None:
        source_index = np.floor(np.arange(target_len) * source_len / target_len).astype(int)
        source_index = np.minimum(source_index, source_len - 1)
        grip = gripper[source_index]
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose.astype(np.float32)


def cubic_spline_resample(delta: np.ndarray, gripper: np.ndarray | None, target_len: int) -> np.ndarray:
    source_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)],
        axis=0,
    )
    target_t = np.linspace(0.0, float(source_len), target_len + 1)
    target_path = CubicSpline(source_t, cumulative, axis=0)(target_t)
    pose = np.diff(target_path, axis=0)

    if gripper is not None:
        source_index = np.floor(np.arange(target_len) * source_len / target_len).astype(int)
        source_index = np.minimum(source_index, source_len - 1)
        grip = gripper[source_index]
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose.astype(np.float32)


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return a_only, b_only, p_value


def evaluate_task_full_horizon(task_cfg: dict, n_episodes: int, task_idx: int, total_tasks: int) -> dict:
    task_name = task_cfg["name"]
    h5_path = task_cfg["h5"]
    json_path = task_cfg["json"]
    control_mode = task_cfg["control_mode"]
    has_gripper = task_cfg.get("has_gripper", True)

    if not os.path.exists(h5_path) or not os.path.exists(json_path):
        return {"name": task_name, "status": "ERROR_FILE_NOT_FOUND"}

    kwargs = {
        "num_envs": 1,
        "obs_mode": "state",
        "render_mode": "rgb_array",
        "control_mode": control_mode,
        "sim_backend": "physx_cpu",
    }
    if task_name != "DrawTriangle-v1":
        kwargs["max_episode_steps"] = 300

    env = gym.make(task_name, **kwargs)

    metadata = json.loads(open(json_path).read())
    episodes = metadata["episodes"][:n_episodes]

    results = {"pchip_fold": [], "cubic_spline": []}

    with h5py.File(h5_path, "r") as h:
        for ep_idx, ep in enumerate(episodes):
            ep_id = int(ep["episode_id"])
            if f"traj_{ep_id}" not in h:
                continue

            actions = np.asarray(h[f"traj_{ep_id}"]["actions"], dtype=np.float32)
            actions = actions[: (len(actions) // 2) * 2]
            executed_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            delta, gripper = typed_coarse(executed_actions, has_gripper)

            target_len = int(len(executed_actions) * 2)  # 40 Hz

            arms = {
                "cubic_spline": cubic_spline_resample(delta, gripper, target_len),
                "pchip_fold": pchip_fold_resample(delta, gripper, target_len),
            }

            state_group = h[f"traj_{ep_id}"]["env_states"]
            state = {
                group: {
                    name: torch.as_tensor(np.asarray(state_group[group][name])[0:1])
                    for name in state_group[group]
                }
                for group in state_group
            }

            for arm_name, acts in arms.items():
                env.reset(seed=ep.get("episode_seed", 0))
                try:
                    env.unwrapped.set_state_dict(state)
                except Exception:
                    pass
                ok = False

                for t in range(len(acts)):
                    try:
                        _, _, term, trunc, info = env.step(acts[t][None])
                        s = info.get("success")
                        if s is not None and bool(np.asarray(s).reshape(-1)[0]):
                            ok = True
                        if bool(np.asarray(term).reshape(-1)[0]):
                            break
                    except Exception:
                        break

                results[arm_name].append(ok)

    env.close()

    pchip_arr = np.asarray(results["pchip_fold"], dtype=bool)
    spline_arr = np.asarray(results["cubic_spline"], dtype=bool)
    pchip_only, spline_only, p_val = exact_mcnemar(pchip_arr, spline_arr)

    pchip_rate = float(np.mean(pchip_arr))
    spline_rate = float(np.mean(spline_arr))
    delta_pp = 100.0 * (pchip_rate - spline_rate)

    return {
        "task_idx": task_idx,
        "task_name": task_name,
        "rate_hz": 40,
        "n_episodes": len(pchip_arr),
        "pchip_success": int(np.sum(pchip_arr)),
        "spline_success": int(np.sum(spline_arr)),
        "pchip_rate": pchip_rate,
        "spline_rate": spline_rate,
        "delta_pp": delta_pp,
        "mcnemar": {
            "pchip_only": pchip_only,
            "spline_only": spline_only,
            "p_value": float(p_val),
        },
    }


def main():
    n_per_task = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    out_file = os.path.join(OUT_DIR, f"maniskill_10tasks_40hz_fullhorizon_results_n{n_per_task}.json")

    print(f"\n=======================================================")
    print(f"EVALUATING 40 HZ FULL HORIZON (n={n_per_task}/task)")
    print(f"=======================================================")

    total_tasks = len(TASKS_CONFIG)
    task_results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_task_full_horizon, cfg, n_per_task, idx, total_tasks): cfg["name"]
            for idx, cfg in enumerate(TASKS_CONFIG, start=1)
        }
        for future in as_completed(futures):
            res = future.result()
            task_results.append(res)
            print(f"[+] Task {res['task_name']} (40 Hz Full-Horizon): PchipFold={res['pchip_rate']*100:.1f}%, Spline={res['spline_rate']*100:.1f}% (Δ = {res['delta_pp']:+.1f}pp, p = {res['mcnemar']['p_value']:.4f})")

    task_results.sort(key=lambda x: x["task_idx"])

    total_pchip_success = sum(t["pchip_success"] for t in task_results)
    total_spline_success = sum(t["spline_success"] for t in task_results)
    total_episodes = sum(t["n_episodes"] for t in task_results)

    agg_pchip_rate = total_pchip_success / total_episodes
    agg_spline_rate = total_spline_success / total_episodes
    agg_delta_pp = (agg_pchip_rate - agg_spline_rate) * 100.0

    scoreboard = {
        "n_per_task": n_per_task,
        "rate_hz": 40,
        "n_episodes_total": total_episodes,
        "aggregate": {
            "pchip_success_total": total_pchip_success,
            "spline_success_total": total_spline_success,
            "pchip_success_rate": agg_pchip_rate,
            "spline_success_rate": agg_spline_rate,
            "delta_pp": agg_delta_pp,
        },
        "task_breakdown": task_results,
    }

    Path(out_file).write_text(json.dumps(scoreboard, indent=2))
    print(f"\n[ALL COMPLETE] 40 Hz Full-Horizon Scoreboard Saved to {out_file}")


if __name__ == "__main__":
    main()
