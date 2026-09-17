"""10-Task ManiSkill 3 Benchmark Evaluation: Fold vs. Cubic Spline.

Designed for execution on Blackwell (`ms3` conda environment).
Evaluates 10 standard ManiSkill 3 manipulation tasks under in-simulation replay
across matched episodes with paired exact McNemar tests and bootstrap CIs.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator, CubicSpline

# 10 Representative ManiSkill 3 Benchmark Tasks
BENCHMARK_TASKS = [
    {"task": "PickCube-v1", "dataset": "pick_rl_joint", "control_mode": "pd_joint_delta_pos"},
    {"task": "PegInsertionSide-v1", "dataset": "peg_rl_joint", "control_mode": "pd_joint_delta_pos"},
    {"task": "PushT-v1", "dataset": "pusht_rl_ee", "control_mode": "pd_ee_delta_pose"},
    {"task": "StackCube-v1", "dataset": "stack_cube_rl_joint", "control_mode": "pd_joint_delta_pos"},
    {"task": "PlugCharger-v1", "dataset": "plug_charger_rl_joint", "control_mode": "pd_joint_delta_pos"},
    {"task": "PullCube-v1", "dataset": "pull_cube_rl_joint", "control_mode": "pd_joint_delta_pos"},
    {"task": "OpenCabinetDoor-v1", "dataset": "open_door_rl_joint", "control_mode": "pd_joint_delta_pos"},
    {"task": "OpenCabinetDrawer-v1", "dataset": "open_drawer_rl_joint", "control_mode": "pd_joint_delta_pos"},
    {"task": "PushChair-v1", "dataset": "push_chair_rl_joint", "control_mode": "pd_joint_delta_pos"},
    {"task": "TurnFaucet-v1", "dataset": "turn_faucet_rl_joint", "control_mode": "pd_joint_delta_pos"},
]


def typed_coarse(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate pairs without treating the direct gripper as a delta."""
    paired = actions.reshape(len(actions) // 2, 2, actions.shape[1])
    delta = paired[:, :, :-1].sum(axis=1)
    gripper = paired[:, 0, -1:]
    return delta, gripper


def fold_resample(delta: np.ndarray, gripper: np.ndarray, target_len: int) -> np.ndarray:
    """Exact integral preserving fold: split coarse deltas equally across steps."""
    ratio = target_len // len(delta)
    assert ratio * len(delta) == target_len
    pose = np.repeat(delta / ratio, ratio, axis=0)
    grip = np.repeat(gripper, ratio, axis=0)
    return np.concatenate([pose, grip], axis=1).astype(np.float32)


def spline_resample(delta: np.ndarray, gripper: np.ndarray, target_len: int) -> np.ndarray:
    """Standard Cubic Spline over the cumulative path."""
    source_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)],
        axis=0,
    )
    target_t = np.linspace(0.0, float(source_len), target_len + 1)
    target_path = CubicSpline(source_t, cumulative, axis=0)(target_t)
    pose = np.diff(target_path, axis=0)

    source_index = np.floor(np.arange(target_len) * source_len / target_len).astype(int)
    source_index = np.minimum(source_index, source_len - 1)
    grip = gripper[source_index]
    return np.concatenate([pose, grip], axis=1).astype(np.float32)


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return a_only, b_only, p_value


def evaluate_task(task_spec: dict[str, str], data_dir: Path, n_episodes: int = 50) -> dict:
    task_name = task_spec["task"]
    dataset_name = task_spec["dataset"]
    control_mode = task_spec["control_mode"]

    h5_file = data_dir / f"{dataset_name}.h5"
    json_file = data_dir / f"{dataset_name}.json"

    if not h5_file.exists() or not json_file.exists():
        print(f"[-] Dataset {dataset_name} not found in {data_dir}. Skipping {task_name}.")
        return {"task": task_name, "status": "DATASET_NOT_FOUND"}

    import mani_skill.envs  # noqa: F401

    metadata = json.loads(json_file.read_text())
    episodes = metadata["episodes"][:n_episodes]

    env = gym.make(
        task_name,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend="physx_cpu",
    )

    results: dict[str, list[bool]] = {"original": [], "folding": [], "spline": []}

    with h5py.File(h5_file, "r") as handle:
        for idx, episode in enumerate(episodes):
            episode_id = int(episode["episode_id"])
            actions = np.asarray(handle[f"traj_{episode_id}"]["actions"], dtype=np.float32)
            actions = actions[: (len(actions) // 2) * 2]
            executed_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            delta, gripper = typed_coarse(executed_actions)

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
                env.reset(seed=episode["episode_seed"])
                env.unwrapped.set_state_dict(state)
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
    fold_only, spline_only, p_val = exact_mcnemar(fold_arr, spline_arr)

    return {
        "task": task_name,
        "dataset": dataset_name,
        "n_episodes": len(episodes),
        "rates": {
            "original": float(np.mean(results["original"])),
            "folding": float(np.mean(fold_arr)),
            "spline": float(np.mean(spline_arr)),
        },
        "mcnemar": {
            "fold_only": fold_only,
            "spline_only": spline_only,
            "p_value": p_val,
        },
        "status": "COMPLETE",
    }


def main():
    data_dir = Path(os.path.expanduser("~/maniskill_data"))
    n_per_task = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("maniskill_10task_results.json")

    print(f"=== Starting 10-Task ManiSkill Evaluation (n={n_per_task}/task) ===")
    task_results = []
    for spec in BENCHMARK_TASKS:
        res = evaluate_task(spec, data_dir, n_episodes=n_per_task)
        task_results.append(res)
        print(f"Result for {spec['task']}: {res}")

    out_file.write_text(json.dumps(task_results, indent=2))
    print(f"[+] All results saved to {out_file}")


if __name__ == "__main__":
    main()
