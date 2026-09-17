"""Parallel 10-Task ManiSkill 3 Evaluation: Fold vs. Spline Replay.

Executes 10 benchmark manipulation tasks in parallel on Blackwell.
Computes paired in-sim success rates, discordant counts, and exact McNemar p-values.
"""

from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from scipy.interpolate import CubicSpline

TASKS_CONFIG = [
    {
        "name": "PickCube-v1",
        "env_id": "PickCube-v1",
        "h5": "/home/user/maniskill_data/pick_rl_joint.h5",
        "json": "/home/user/maniskill_data/pick_rl_joint.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "LiftPegUpright-v1",
        "env_id": "LiftPegUpright-v1",
        "h5": "/home/user/maniskill_data/raw/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PegInsertionSide-v1",
        "env_id": "PegInsertionSide-v1",
        "h5": "/home/user/maniskill_data/raw/PegInsertionSide-v1/rl/trajectory.h5",
        "json": "/home/user/maniskill_data/raw/PegInsertionSide-v1/rl/trajectory.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "StackCube-v1",
        "env_id": "StackCube-v1",
        "h5": "/home/user/maniskill_data/raw/StackCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/StackCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PullCube-v1",
        "env_id": "PullCube-v1",
        "h5": "/home/user/maniskill_data/raw/PullCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/PullCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PokeCube-v1",
        "env_id": "PokeCube-v1",
        "h5": "/home/user/maniskill_data/raw/PokeCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/PokeCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PushCube-v1",
        "env_id": "PushCube-v1",
        "h5": "/home/user/maniskill_data/raw/PushCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/PushCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "RollBall-v1",
        "env_id": "RollBall-v1",
        "h5": "/home/user/maniskill_data/raw/RollBall-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/RollBall-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "DrawTriangle-v1",
        "env_id": "DrawTriangle-v1",
        "h5": "/home/user/maniskill_data/raw/DrawTriangle-v1/motionplanning/trajectory.h5",
        "json": "/home/user/maniskill_data/raw/DrawTriangle-v1/motionplanning/trajectory.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "PushT-v1",
        "env_id": "PushT-v1",
        "h5": "/home/user/maniskill_data/pusht_rl.h5",
        "json": "/home/user/maniskill_data/pusht_rl.json",
        "control_mode": "pd_ee_delta_pose",
        "has_gripper": True,
    },
]


def typed_coarse(actions: np.ndarray, has_gripper: bool) -> tuple[np.ndarray, np.ndarray | None]:
    paired = actions.reshape(len(actions) // 2, 2, actions.shape[1])
    if has_gripper:
        delta = paired[:, :, :-1].sum(axis=1)
        gripper = paired[:, 0, -1:]
        return delta, gripper
    else:
        delta = paired.sum(axis=1)
        return delta, None


def fold_resample(delta: np.ndarray, gripper: np.ndarray | None, target_len: int) -> np.ndarray:
    ratio = target_len // len(delta)
    pose = np.repeat(delta / ratio, ratio, axis=0)
    if gripper is not None:
        grip = np.repeat(gripper, ratio, axis=0)
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose.astype(np.float32)


def spline_resample(delta: np.ndarray, gripper: np.ndarray | None, target_len: int) -> np.ndarray:
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


def evaluate_single_task(cfg: dict, n_episodes: int = 50) -> dict:
    name = cfg["name"]
    env_id = cfg["env_id"]
    h5_path = Path(cfg["h5"])
    json_path = Path(cfg["json"])
    control_mode = cfg["control_mode"]
    has_gripper = cfg.get("has_gripper", True)

    if not h5_path.exists() or not json_path.exists():
        return {"name": name, "status": "ERROR_FILE_NOT_FOUND", "path": str(h5_path)}

    import mani_skill.envs  # noqa: F401

    try:
        metadata = json.loads(json_path.read_text())
        episodes = metadata["episodes"][:n_episodes]

        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="state",
            control_mode=control_mode,
            sim_backend="physx_cpu",
        )

        results: dict[str, list[bool]] = {"original": [], "folding": [], "spline": []}

        with h5py.File(h5_path, "r") as handle:
            for idx, episode in enumerate(episodes):
                episode_id = int(episode["episode_id"])
                if f"traj_{episode_id}" not in handle:
                    continue
                actions = np.asarray(handle[f"traj_{episode_id}"]["actions"], dtype=np.float32)
                actions = actions[: (len(actions) // 2) * 2]
                executed_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
                delta, gripper = typed_coarse(executed_actions, has_gripper)

                arms = {
                    "original": executed_actions,
                    "folding": fold_resample(delta, gripper, len(executed_actions)),
                    "spline": spline_resample(delta, gripper, len(executed_actions)),
                }

                state_group = handle[f"traj_{episode_id}"]["env_states"]
                state = {
                    group: {
                        name: torch.as_tensor(np.asarray(state_group[group][name])[0:1])
                        for name in state_group[group]
                    }
                    for group in state_group
                }

                for arm, arm_actions in arms.items():
                    env.reset(seed=episode.get("episode_seed", 0))
                    try:
                        env.unwrapped.set_state_dict(state)
                    except Exception:
                        pass
                    succeeded = False
                    for action in arm_actions:
                        _, _, terminated, truncated, info = env.step(action[None])
                        success = info.get("success")
                        if success is not None and bool(np.asarray(success).reshape(-1)[0]):
                            succeeded = True
                        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                            np.asarray(truncated).reshape(-1)[0]
                        ):
                            break
                    results[arm].append(succeeded)

        env.close()

        fold_arr = np.asarray(results["folding"], dtype=bool)
        spline_arr = np.asarray(results["spline"], dtype=bool)
        orig_arr = np.asarray(results["original"], dtype=bool)

        n_actual = len(orig_arr)
        if n_actual == 0:
            return {"name": name, "status": "ERROR_NO_VALID_EPISODES"}

        fold_only, spline_only, p_val = exact_mcnemar(fold_arr, spline_arr)

        return {
            "name": name,
            "status": "SUCCESS",
            "n_episodes": n_actual,
            "rates": {
                "original": float(np.mean(orig_arr)),
                "folding": float(np.mean(fold_arr)),
                "spline": float(np.mean(spline_arr)),
            },
            "counts": {
                "original": int(np.sum(orig_arr)),
                "folding": int(np.sum(fold_arr)),
                "spline": int(np.sum(spline_arr)),
            },
            "comparison": {
                "fold_minus_spline_pp": float(100.0 * (np.mean(fold_arr) - np.mean(spline_arr))),
                "fold_only": fold_only,
                "spline_only": spline_only,
                "p_value": float(p_val),
            },
        }

    except Exception as e:
        return {"name": name, "status": "EXCEPTION", "error": str(e)}


def main():
    n_per_task = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("maniskill_10tasks_parallel_results.json")
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    print(f"=== Running 10-Task ManiSkill Benchmark Parallel Evaluation (n={n_per_task}/task, workers={max_workers}) ===")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_single_task, cfg, n_per_task): cfg["name"] for cfg in TASKS_CONFIG}
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                res = future.result()
                all_results.append(res)
                print(f"[+] Task {task_name} finished: {res.get('status')}")
                if res.get("status") == "SUCCESS":
                    print(f"    Orig: {res['rates']['original']*100:.1f}%, Fold: {res['rates']['folding']*100:.1f}%, Spline: {res['rates']['spline']*100:.1f}% (Δ = {res['comparison']['fold_minus_spline_pp']:+.1f}pp, p = {res['comparison']['p_value']:.4f})")
            except Exception as exc:
                print(f"[-] Task {task_name} generated exception: {exc}")

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[DONE] Evaluated {len(all_results)} tasks. Summary saved to {out_path}")


if __name__ == "__main__":
    main()
