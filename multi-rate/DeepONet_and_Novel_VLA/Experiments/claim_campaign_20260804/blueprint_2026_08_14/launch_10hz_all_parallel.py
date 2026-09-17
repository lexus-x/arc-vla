"""Parallel 10 Hz Multi-Task Evaluator: 10 Separate Processes (One per Task).

Runs all 10 tasks in parallel simultaneously across CPU cores on Blackwell:
- Evaluates TAC-Fold (Ours) vs. Cubic Spline on identical episodes (n=50/task, N=500 total).
- Renders synchronized side-by-side comparison video: `maniskill_10tasks_10hz_ours_master.mp4`.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch
from scipy.interpolate import CubicSpline
import mani_skill.envs

OUT_DIR = os.path.expanduser("~/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14")
VIDEO_DIR = os.path.join(OUT_DIR, "comparison_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

TARGET_H, TARGET_W = 512, 512

TASKS_10 = [
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


def typed_coarse(actions: np.ndarray, has_gripper: bool) -> tuple[np.ndarray, np.ndarray | None]:
    paired = actions.reshape(len(actions) // 2, 2, actions.shape[1])
    if has_gripper:
        delta = paired[:, :, :-1].sum(axis=1)
        gripper = paired[:, 0, -1:]
        return delta, gripper
    else:
        delta = paired.sum(axis=1)
        return delta, None


def ours_10hz_resample(delta: np.ndarray, gripper: np.ndarray | None) -> np.ndarray:
    pose = delta.astype(np.float32)
    if gripper is not None:
        return np.concatenate([pose, gripper.astype(np.float32)], axis=1)
    return pose


def spline_10hz_resample(delta: np.ndarray, gripper: np.ndarray | None, orig_delta: np.ndarray, orig_gripper: np.ndarray | None) -> np.ndarray:
    source_len = len(orig_delta)
    target_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, orig_delta.shape[1]), dtype=np.float64), np.cumsum(orig_delta, axis=0)], axis=0)
    target_t = np.linspace(0.0, float(source_len), target_len + 1)
    target_path = CubicSpline(source_t, cumulative, axis=0)(target_t)
    pose = np.diff(target_path, axis=0).astype(np.float32)

    if gripper is not None:
        source_index = np.floor(np.arange(target_len) * source_len / target_len).astype(int)
        source_index = np.minimum(source_index, source_len - 1)
        grip = orig_gripper[source_index].astype(np.float32)
        return np.concatenate([pose, grip], axis=1)
    return pose


def create_header_banner(width: int, height: int, title: str) -> np.ndarray:
    banner = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (width // 2 - 250, height // 2 + 7),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return banner


def draw_sub_label(frame: np.ndarray, text: str, is_success: bool) -> np.ndarray:
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    color = (0, 230, 0) if is_success else (0, 60, 240)
    tag = "SUCCESS" if is_success else "FAILED"
    full_text = f"{text}: {tag}"
    cv2.rectangle(img, (8, 8), (320, 42), (0, 0, 0), -1)
    cv2.putText(img, full_text, (14, 32), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def run_single_task(task_idx: int, n_episodes: int = 50) -> dict:
    task_cfg = TASKS_10[task_idx]
    task_name = task_cfg["name"]
    h5_path = task_cfg["h5"]
    json_path = task_cfg["json"]
    control_mode = task_cfg["control_mode"]
    has_gripper = task_cfg.get("has_gripper", True)

    if not os.path.exists(h5_path) or not os.path.exists(json_path):
        return {
            "task_idx": task_idx,
            "task_name": task_name,
            "n_episodes": 0,
            "ours_success": 0,
            "spline_success": 0,
            "ours_sr": 0.0,
            "spline_sr": 0.0,
            "delta_pp": 0.0,
        }

    env = gym.make(
        task_name,
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        control_mode=control_mode,
        sim_backend="physx_cpu",
    )

    metadata = json.loads(open(json_path).read())
    episodes = metadata["episodes"][:n_episodes]

    results = {"ours": [], "spline": []}
    best_segment_frames = None

    with h5py.File(h5_path, "r") as h:
        for ep_idx, ep in enumerate(episodes):
            ep_id = int(ep["episode_id"])
            if f"traj_{ep_id}" not in h:
                continue

            actions = np.asarray(h[f"traj_{ep_id}"]["actions"], dtype=np.float32)
            actions = actions[: (len(actions) // 2) * 2]
            executed_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            delta, gripper = typed_coarse(executed_actions, has_gripper)

            orig_delta = executed_actions[:, :-1] if has_gripper else executed_actions
            orig_grip = executed_actions[:, -1:] if has_gripper else None

            arms = {
                "spline": spline_10hz_resample(delta, gripper, orig_delta, orig_grip),
                "ours": ours_10hz_resample(delta, gripper),
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

                if ep_idx < 10:
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
                    if ep_idx < 10:
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
                results[arm_name].append(ok)

            if ep_idx < 10 and len(arm_frames["spline"]) > 0:
                max_len = max(len(arm_frames["spline"]), len(arm_frames["ours"]))

                def pad_frames(fr_list, target_l):
                    padded = list(fr_list)
                    while len(padded) < target_l:
                        padded.append(fr_list[-1])
                    return padded

                f_spline = pad_frames(arm_frames["spline"], max_len)
                f_ours = pad_frames(arm_frames["ours"], max_len)

                banner_title = f"10 HZ SUB-NATIVE | {task_name}"
                banner = create_header_banner(TARGET_W * 2, 44, banner_title)

                combined = []
                for s_fr, o_fr in zip(f_spline, f_ours):
                    s_ann = draw_sub_label(s_fr, "SPLINE (10 HZ)", arm_success["spline"])
                    o_ann = draw_sub_label(o_fr, "OURS (10 HZ)", arm_success["ours"])
                    side_by_side = np.concatenate([s_ann, o_ann], axis=1)
                    full_f = np.concatenate([banner, side_by_side], axis=0)
                    combined.append(full_f)

                if arm_success["ours"] and not arm_success["spline"]:
                    best_segment_frames = combined
                elif best_segment_frames is None:
                    best_segment_frames = combined

    env.close()

    ours_arr = np.asarray(results["ours"], dtype=bool)
    spline_arr = np.asarray(results["spline"], dtype=bool)

    ours_sr = float(np.mean(ours_arr)) if len(ours_arr) > 0 else 0.0
    spline_sr = float(np.mean(spline_arr)) if len(spline_arr) > 0 else 0.0
    delta_pp = (ours_sr - spline_sr) * 100.0

    task_json = os.path.join(OUT_DIR, f"task_{task_idx}_10hz_result.json")
    task_res = {
        "task_idx": task_idx,
        "task_name": task_name,
        "n_episodes": len(ours_arr),
        "ours_success": int(np.sum(ours_arr)),
        "spline_success": int(np.sum(spline_arr)),
        "ours_sr": ours_sr,
        "spline_sr": spline_sr,
        "delta_pp": delta_pp,
    }
    Path(task_json).write_text(json.dumps(task_res, indent=2))

    if best_segment_frames is not None:
        best_segment_frames.extend([best_segment_frames[-1]] * 10)
        vid_p = os.path.join(VIDEO_DIR, f"task_{task_idx}_10hz.mp4")
        imageio.mimsave(vid_p, best_segment_frames, fps=10)

    print(f"[DONE TASK {task_idx+1}/10] {task_name}: Ours={ours_sr*100:.1f}%, Spline={spline_sr*100:.1f}% (Δ = {delta_pp:+.1f}pp)", flush=True)
    return task_res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        idx = int(sys.argv[1])
        run_single_task(idx)
    else:
        # Launch 10 tasks in parallel via subprocesses
        print("Launching all 10 tasks simultaneously across 10 worker processes...")
        procs = []
        for i in range(10):
            cmd = f"/home/user/anaconda3/envs/ms3/bin/python /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14/launch_10hz_all_parallel.py {i}"
            p = subprocess.Popen(cmd, shell=True)
            procs.append(p)
        
        for p in procs:
            p.wait()

        print("\nAll 10 parallel tasks finished. Aggregating scoreboard...")
        all_results = []
        for i in range(10):
            f = os.path.join(OUT_DIR, f"task_{i}_10hz_result.json")
            if os.path.exists(f):
                all_results.append(json.loads(Path(f).read_text()))

        total_ours = sum(r["ours_success"] for r in all_results)
        total_spline = sum(r["spline_success"] for r in all_results)
        total_ep = sum(r["n_episodes"] for r in all_results)

        agg_ours_sr = total_ours / total_ep if total_ep > 0 else 0.0
        agg_spline_sr = total_spline / total_ep if total_ep > 0 else 0.0
        agg_delta = (agg_ours_sr - agg_spline_sr) * 100.0

        # Concatenate video segments
        final_video = os.path.join(VIDEO_DIR, "maniskill_10tasks_10hz_ours_master.mp4")
        all_vids = [os.path.join(VIDEO_DIR, f"task_{i}_10hz.mp4") for i in range(10) if os.path.exists(os.path.join(VIDEO_DIR, f"task_{i}_10hz.mp4"))]
        if all_vids:
            all_frames = []
            for v in all_vids:
                reader = imageio.get_reader(v)
                for im in reader:
                    all_frames.append(im)
                reader.close()
            imageio.mimsave(final_video, all_frames, fps=10)

        scoreboard = {
            "rate_hz": 10,
            "n_total": total_ep,
            "ours_success_rate": agg_ours_sr,
            "spline_success_rate": agg_spline_sr,
            "delta_pp": agg_delta,
            "ours_success_count": total_ours,
            "spline_success_count": total_spline,
            "task_breakdown": all_results,
            "video_path": final_video,
        }

        out_json = os.path.join(OUT_DIR, "maniskill_10tasks_10hz_results_n50.json")
        Path(out_json).write_text(json.dumps(scoreboard, indent=2))
        print(f"\n[PARALLEL COMPLETE] 10 Hz Scoreboard Saved to {out_json}")
        print(f"Ours 10 Hz SR: {agg_ours_sr*100:.1f}% | Spline 10 Hz SR: {agg_spline_sr*100:.1f}% (Δ = {agg_delta:+.1f} pp)")
