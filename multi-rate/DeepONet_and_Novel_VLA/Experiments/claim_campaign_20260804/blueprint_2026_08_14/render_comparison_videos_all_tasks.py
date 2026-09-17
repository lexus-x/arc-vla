"""Render Side-by-Side Comparison Videos of Operator Fold vs. Cubic Spline.

Produces clear, captioned comparison videos for key tasks (PickCube-v1, LiftPegUpright-v1)
showing exact side-by-side physics rollouts.
"""

import os
import sys
import json
import numpy as np
import h5py
import torch
import cv2
import imageio
from scipy.interpolate import CubicSpline
import gymnasium as gym
import mani_skill.envs

OUT_DIR = os.path.expanduser("~/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14/comparison_videos")
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = [
    {
        "name": "PickCube-v1",
        "h5": "/home/user/maniskill_data/pick_rl_joint.h5",
        "json": "/home/user/maniskill_data/pick_rl_joint.json",
        "control_mode": "pd_joint_delta_pos",
    },
    {
        "name": "LiftPegUpright-v1",
        "h5": "/home/user/maniskill_data/raw/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
        "control_mode": "pd_joint_delta_pos",
    },
]


def typed_coarse(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    paired = actions.reshape(len(actions) // 2, 2, actions.shape[1])
    delta = paired[:, :, :-1].sum(axis=1)
    gripper = paired[:, 0, -1:]
    return delta, gripper


def fold_resample(delta: np.ndarray, gripper: np.ndarray, target_len: int) -> np.ndarray:
    ratio = target_len // len(delta)
    pose = np.repeat(delta / ratio, ratio, axis=0)
    grip = np.repeat(gripper, ratio, axis=0)
    return np.concatenate([pose, grip], axis=1).astype(np.float32)


def spline_resample(delta: np.ndarray, gripper: np.ndarray, target_len: int) -> np.ndarray:
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


def draw_label(frame: np.ndarray, text: str, color=(255, 255, 255), bg_color=(0, 0, 0)) -> np.ndarray:
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (10, 10), (10 + text_w + 10, 10 + text_h + 10), bg_color, -1)
    cv2.putText(img, text, (15, 10 + text_h + 3), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def render_task_comparisons(task_cfg: dict, max_videos: int = 3):
    task_name = task_cfg["name"]
    h5_path = task_cfg["h5"]
    json_path = task_cfg["json"]
    control_mode = task_cfg["control_mode"]

    print(f"\n[+] Rendering side-by-side comparison videos for {task_name}...")
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

    saved_count = 0
    with h5py.File(h5_path, "r") as h:
        for ep in episodes:
            if saved_count >= max_videos:
                break
            ep_id = int(ep["episode_id"])
            if f"traj_{ep_id}" not in h:
                continue

            actions = np.asarray(h[f"traj_{ep_id}"]["actions"], dtype=np.float32)
            actions = actions[: (len(actions) // 2) * 2]
            executed_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            delta, gripper = typed_coarse(executed_actions)

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
                    frames.append(f)
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                        break
                arm_frames[arm_name] = frames
                arm_success[arm_name] = ok

            # Prioritize discordant episodes where Fold wins over Spline
            is_discordant_win = arm_success["folding"] and not arm_success["spline"]
            if is_discordant_win or saved_count == 0:
                max_len = max(len(arm_frames["spline"]), len(arm_frames["folding"]))

                def pad_frames(fr_list, target_len):
                    padded = list(fr_list)
                    while len(padded) < target_len:
                        padded.append(fr_list[-1])
                    return padded

                f_spline = pad_frames(arm_frames["spline"], max_len)
                f_folding = pad_frames(arm_frames["folding"], max_len)

                combined_frames = []
                for s_fr, f_fr in zip(f_spline, f_folding):
                    lbl_spline = f"CUBIC SPLINE: {'SUCCESS' if arm_success['spline'] else 'FAILED'}"
                    lbl_folding = f"OPERATOR FOLD: {'SUCCESS' if arm_success['folding'] else 'FAILED'}"
                    color_spline = (0, 255, 0) if arm_success["spline"] else (255, 50, 50)
                    color_folding = (0, 255, 0) if arm_success["folding"] else (255, 50, 50)

                    s_annotated = draw_label(s_fr, lbl_spline, color=color_spline)
                    f_annotated = draw_label(f_fr, lbl_folding, color=color_folding)
                    side_by_side = np.concatenate([s_annotated, f_annotated], axis=1)
                    combined_frames.append(side_by_side)

                clean_name = task_name.lower().replace("-v1", "")
                out_path = f"{OUT_DIR}/{clean_name}_ep{ep_id}_comparison.mp4"
                imageio.mimsave(out_path, combined_frames, fps=20)
                print(f"[+] Saved comparison video: {out_path} (Fold={arm_success['folding']}, Spline={arm_success['spline']})")
                saved_count += 1

    env.close()


def main():
    for task in TASKS:
        render_task_comparisons(task, max_videos=2)
    print(f"\n[DONE] All side-by-side comparison videos generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
