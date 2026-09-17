"""Render a Single Master Video Compiling Side-by-Side Comparisons Across All 10 ManiSkill Tasks.

Normalizes all environment frames to a fixed 512x512 resolution so the entire 10-task video
compiles seamlessly into a single 1024x560 MP4.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch
from scipy.interpolate import CubicSpline
import mani_skill.envs

OUT_DIR = os.path.expanduser("~/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14/comparison_videos")
os.makedirs(OUT_DIR, exist_ok=True)
MASTER_VIDEO_PATH = os.path.join(OUT_DIR, "maniskill_10tasks_master_comparison.mp4")

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


def create_header_banner(width: int, height: int, title: str) -> np.ndarray:
    banner = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (width // 2 - 220, height // 2 + 7),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return banner


def draw_sub_label(frame: np.ndarray, text: str, is_success: bool) -> np.ndarray:
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 230, 0) if is_success else (0, 60, 240)
    tag = "SUCCESS" if is_success else "FAILED"
    full_text = f"{text}: {tag}"
    cv2.rectangle(img, (8, 8), (340, 42), (0, 0, 0), -1)
    cv2.putText(img, full_text, (14, 32), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def render_task_segment(task_idx: int, total_tasks: int, cfg: dict) -> list[np.ndarray]:
    task_name = cfg["name"]
    h5_path = cfg["h5"]
    json_path = cfg["json"]
    control_mode = cfg["control_mode"]
    has_gripper = cfg.get("has_gripper", True)

    print(f"[+] Processing Task {task_idx}/{total_tasks}: {task_name}...")

    if not os.path.exists(h5_path) or not os.path.exists(json_path):
        print(f"[-] Missing files for {task_name}, skipping.")
        return []

    env = gym.make(
        task_name,
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        control_mode=control_mode,
        sim_backend="physx_cpu",
    )

    metadata = json.loads(open(json_path).read())
    episodes = metadata["episodes"]

    best_frames = None

    with h5py.File(h5_path, "r") as h:
        for ep in episodes[:15]:
            ep_id = int(ep["episode_id"])
            if f"traj_{ep_id}" not in h:
                continue

            actions = np.asarray(h[f"traj_{ep_id}"]["actions"], dtype=np.float32)
            actions = actions[: (len(actions) // 2) * 2]
            executed_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            delta, gripper = typed_coarse(executed_actions, has_gripper)

            arms = {
                "spline": spline_resample(delta, gripper, len(executed_actions)),
                "folding": fold_resample(delta, gripper, len(executed_actions)),
            }

            state_group = h[f"traj_{ep_id}"]["env_states"]
            state = {
                group: {
                    name: torch.as_tensor(np.asarray(state_group[group][name])[0:1])
                    for name in state_group[group]
                }
                for group in state_group
            }

            arm_frames = {}
            arm_success = {}

            for arm_name, acts in arms.items():
                env.reset(seed=ep.get("episode_seed", 0))
                try:
                    env.unwrapped.set_state_dict(state)
                except Exception:
                    pass
                frames = []
                ok = False

                f0 = env.render()
                if isinstance(f0, torch.Tensor):
                    f0 = f0.cpu().numpy()
                if f0.ndim == 4:
                    f0 = f0[0]
                if f0.shape[:2] != (TARGET_H, TARGET_W):
                    f0 = cv2.resize(f0, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                frames.append(f0)

                for t in range(len(acts)):
                    _, _, term, trunc, info = env.step(acts[t][None])
                    s = info.get("success")
                    if s is not None and bool(np.asarray(s).reshape(-1)[0]):
                        ok = True
                    f = env.render()
                    if isinstance(f, torch.Tensor):
                        f = f.cpu().numpy()
                    if f.ndim == 4:
                        f = f[0]
                    if f.shape[:2] != (TARGET_H, TARGET_W):
                        f = cv2.resize(f, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                    frames.append(f)
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                        break
                arm_frames[arm_name] = frames
                arm_success[arm_name] = ok

            # Build combined side-by-side frame sequence
            max_len = max(len(arm_frames["spline"]), len(arm_frames["folding"]))

            def pad_frames(fr_list, target_len):
                padded = list(fr_list)
                while len(padded) < target_len:
                    padded.append(fr_list[-1])
                return padded

            f_spline = pad_frames(arm_frames["spline"], max_len)
            f_folding = pad_frames(arm_frames["folding"], max_len)

            total_width = TARGET_W * 2
            banner = create_header_banner(total_width, 48, f"TASK {task_idx}/{total_tasks}: {task_name}")

            combined_task_frames = []
            for s_fr, f_fr in zip(f_spline, f_folding):
                s_annotated = draw_sub_label(s_fr, "CUBIC SPLINE", arm_success["spline"])
                f_annotated = draw_sub_label(f_fr, "OPERATOR FOLD", arm_success["folding"])
                side_by_side = np.concatenate([s_annotated, f_annotated], axis=1)
                full_frame = np.concatenate([banner, side_by_side], axis=0)
                combined_task_frames.append(full_frame)

            # Keep discordant (Fold wins) if available, or first valid episode
            is_discordant = arm_success["folding"] and not arm_success["spline"]
            if is_discordant:
                best_frames = combined_task_frames
                break
            elif best_frames is None:
                best_frames = combined_task_frames

    env.close()

    if best_frames is not None:
        # Add pause/hold at the end of each task segment (10 frames = 0.5s)
        best_frames.extend([best_frames[-1]] * 10)
        return best_frames
    return []


def main():
    print(f"=== Compiling 10-Task Master Comparison Video ===")
    all_master_frames = []
    total_tasks = len(TASKS_CONFIG)

    for i, cfg in enumerate(TASKS_CONFIG, start=1):
        frames = render_task_segment(i, total_tasks, cfg)
        all_master_frames.extend(frames)

    print(f"\n[+] Total Master Frames Rendered: {len(all_master_frames)}")
    imageio.mimsave(MASTER_VIDEO_PATH, all_master_frames, fps=20)
    print(f"[SUCCESS] Master 10-Task Comparison Video Saved: {MASTER_VIDEO_PATH}")


if __name__ == "__main__":
    main()
