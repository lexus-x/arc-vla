"""Paired ManiSkill replay: typed equal-split folding vs cumulative PCHIP.

Run on Blackwell only. The first action_dim - 1 channels are delta commands;
the last channel is a direct gripper command and is held causally.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator


def typed_coarse(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate pairs without treating the direct gripper as a delta."""
    paired = actions.reshape(len(actions) // 2, 2, actions.shape[1])
    delta = paired[:, :, :-1].sum(axis=1)
    gripper = paired[:, 0, -1:]
    return delta, gripper


def equal_split(delta: np.ndarray, gripper: np.ndarray, target_len: int) -> np.ndarray:
    ratio = target_len // len(delta)
    assert ratio * len(delta) == target_len
    pose = np.repeat(delta / ratio, ratio, axis=0)
    grip = np.repeat(gripper, ratio, axis=0)
    return np.concatenate([pose, grip], axis=1).astype(np.float32)


def cumulative_pchip(
    delta: np.ndarray, gripper: np.ndarray, target_len: int
) -> np.ndarray:
    """Interpolate cumulative delta path, then difference at target boundaries."""
    source_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)],
        axis=0,
    )
    target_t = np.linspace(0.0, float(source_len), target_len + 1)
    target_path = PchipInterpolator(source_t, cumulative, axis=0)(target_t)
    pose = np.diff(target_path, axis=0)

    source_index = np.floor(np.arange(target_len) * source_len / target_len).astype(int)
    source_index = np.minimum(source_index, source_len - 1)
    grip = gripper[source_index]

    assert np.isfinite(pose).all()
    assert np.allclose(pose.sum(axis=0), delta.sum(axis=0), atol=1e-6, rtol=1e-6)
    assert set(np.unique(grip)).issubset(set(np.unique(gripper)))
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


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260814)
    differences = a.astype(np.float64) - b.astype(np.float64)
    draws = rng.choice(differences, size=(10_000, len(differences)), replace=True)
    return tuple(float(x) for x in np.quantile(draws.mean(axis=1), [0.025, 0.975]))


def main() -> None:
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "typed_pchip_replay_n300.json"
    )
    data_dir = Path(os.path.expanduser("~/maniskill_data"))
    metadata = json.loads((data_dir / "pick_rl_joint.json").read_text())
    episodes = metadata["episodes"][:n_episodes]

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
    )
    results: dict[str, list[bool]] = {"original": [], "folding": [], "pchip": []}
    episode_ids: list[int] = []

    with h5py.File(data_dir / "pick_rl_joint.h5", "r") as handle:
        for index, episode in enumerate(episodes):
            episode_id = int(episode["episode_id"])
            actions = np.asarray(
                handle[f"traj_{episode_id}"]["actions"], dtype=np.float32
            )
            actions = actions[: (len(actions) // 2) * 2]
            # ManiSkill normalizes this controller's action space and clips every
            # command to [-1, 1] before scaling. Coarsen the commands that were
            # actually executed, not the unbounded values stored in the dataset.
            executed_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            delta, gripper = typed_coarse(executed_actions)
            arms = {
                "original": executed_actions,
                "folding": equal_split(delta, gripper, len(executed_actions)),
                "pchip": cumulative_pchip(delta, gripper, len(executed_actions)),
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

            episode_ids.append(episode_id)
            if (index + 1) % 25 == 0 or index == 0:
                print(f"completed {index + 1}/{len(episodes)}", flush=True)

    env.close()
    fold = np.asarray(results["folding"], dtype=bool)
    pchip = np.asarray(results["pchip"], dtype=bool)
    fold_only, pchip_only, p_value = exact_mcnemar(fold, pchip)
    ci_low, ci_high = paired_bootstrap_ci(fold, pchip)

    summary = {
        arm: {
            "successes": int(sum(values)),
            "total": len(values),
            "rate": float(np.mean(values)),
        }
        for arm, values in results.items()
    }
    payload = {
        "protocol": {
            "dataset": "pick_rl_joint",
            "environment": "PickCube-v1",
            "control_mode": "pd_joint_delta_pos",
            "paired_episodes": len(episodes),
            "delta_channels": "0:-1",
            "gripper": "last channel, causal ZOH from first action in each pair",
            "coarsening_space": "controller-normalized commands clipped to [-1, 1]",
            "primary_contrast": "folding_vs_cumulative_pchip",
            "bootstrap_seed": 20260814,
        },
        "summary": summary,
        "primary": {
            "delta_percentage_points": 100.0 * (float(fold.mean()) - float(pchip.mean())),
            "folding_only": fold_only,
            "pchip_only": pchip_only,
            "exact_mcnemar_p": p_value,
            "paired_bootstrap_95ci_percentage_points": [100.0 * ci_low, 100.0 * ci_high],
        },
        "episode_ids": episode_ids,
        "outcomes": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"summary": summary, "primary": payload["primary"]}, indent=2))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
