"""Multi-Task Multi-Rate Parallel Evaluation (n=100 per task, n=1,000 per rate).

Evaluates all 10 ManiSkill benchmark tasks at:
- 40 Hz (2x upsampling) -> Produces `maniskill_10tasks_40hz_master_comparison.mp4`
- 10 Hz (2x decimation) -> Produces `maniskill_10tasks_10hz_master_comparison.mp4`

Executes in parallel across 8 workers on Blackwell.
Only saves the full 10-task master comparison videos.
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
from scipy.interpolate import CubicSpline
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


def fold_resample(delta: np.ndarray, gripper: np.ndarray | None, target_len: int) -> np.ndarray:
    if target_len >= len(delta):
        ratio = target_len // len(delta)
        pose = np.repeat(delta / ratio, ratio, axis=0)
        if gripper is not None:
            grip = np.repeat(gripper, ratio, axis=0)
            return np.concatenate([pose, grip], axis=1).astype(np.float32)
        return pose.astype(np.float32)
    else:
        k = len(delta) // target_len
        pose = delta.reshape(target_len, k, delta.shape[1]).sum(axis=1)
        if gripper is not None:
            grip = gripper[::k][:target_len]
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
        (width // 2 - 250, height // 2 + 7),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
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


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return a_only, b_only, p_value


def evaluate_task_at_rate(task_cfg: dict, target_rate: int, n_episodes: int, task_idx: int, total_tasks: int) -> dict:
    task_name = task_cfg["name"]
    h5_path = task_cfg["h5"]
    json_path = task_cfg["json"]
    control_mode = task_cfg["control_mode"]
    has_gripper = task_cfg.get("has_gripper", True)

    if not os.path.exists(h5_path) or not os.path.exists(json_path):
        return {"name": task_name, "status": "ERROR_FILE_NOT_FOUND"}

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

    results = {"folding": [], "spline": []}
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

            target_len = max(2, int(len(executed_actions) * (target_rate / 20.0)))

            arms = {
                "spline": spline_resample(delta, gripper, target_len),
                "folding": fold_resample(delta, gripper, target_len),
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
                max_len = max(len(arm_frames["spline"]), len(arm_frames["folding"]))

                def pad_frames(fr_list, target_l):
                    padded = list(fr_list)
                    while len(padded) < target_l:
                        padded.append(fr_list[-1])
                    return padded

                f_spline = pad_frames(arm_frames["spline"], max_len)
                f_folding = pad_frames(arm_frames["folding"], max_len)

                total_width = TARGET_W * 2
                banner_title = f"{target_rate} HZ | TASK {task_idx}/{total_tasks}: {task_name}"
                banner = create_header_banner(total_width, 48, banner_title)

                combined_task_frames = []
                for s_fr, f_fr in zip(f_spline, f_folding):
                    s_ann = draw_sub_label(s_fr, "CUBIC SPLINE", arm_success["spline"])
                    f_ann = draw_sub_label(f_fr, "OPERATOR FOLD", arm_success["folding"])
                    side_by_side = np.concatenate([s_ann, f_ann], axis=1)
                    full_f = np.concatenate([banner, side_by_side], axis=0)
                    combined_task_frames.append(full_f)

                is_discordant = arm_success["folding"] and not arm_success["spline"]
                if is_discordant:
                    best_segment_frames = combined_task_frames
                elif best_segment_frames is None:
                    best_segment_frames = combined_task_frames

    env.close()

    fold_arr = np.asarray(results["folding"], dtype=bool)
    spline_arr = np.asarray(results["spline"], dtype=bool)
    fold_only, spline_only, p_val = exact_mcnemar(fold_arr, spline_arr)

    fold_rate = float(np.mean(fold_arr))
    spline_rate = float(np.mean(spline_arr))
    delta_pp = 100.0 * (fold_rate - spline_rate)

    if best_segment_frames is not None:
        best_segment_frames.extend([best_segment_frames[-1]] * 10)

    return {
        "task_idx": task_idx,
        "task_name": task_name,
        "rate_hz": target_rate,
        "n_episodes": len(fold_arr),
        "folding_success": int(np.sum(fold_arr)),
        "spline_success": int(np.sum(spline_arr)),
        "folding_rate": fold_rate,
        "spline_rate": spline_rate,
        "delta_pp": delta_pp,
        "mcnemar": {
            "fold_only": fold_only,
            "spline_only": spline_only,
            "p_value": float(p_val),
        },
        "frames": best_segment_frames,
    }


def run_rate_suite(target_rate: int, n_per_task: int = 100, max_workers: int = 8) -> dict:
    print(f"\n=======================================================")
    print(f"RUNNING PARALLEL 10-TASK SUITE AT {target_rate} HZ (n={n_per_task}/task)")
    print(f"=======================================================")

    total_tasks = len(TASKS_CONFIG)
    task_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_task_at_rate, cfg, target_rate, n_per_task, idx, total_tasks): cfg["name"]
            for idx, cfg in enumerate(TASKS_CONFIG, start=1)
        }
        for future in as_completed(futures):
            res = future.result()
            task_results.append(res)
            print(f"[+] Task {res['task_name']} ({target_rate} Hz): Fold={res['folding_rate']*100:.1f}%, Spline={res['spline_rate']*100:.1f}% (Δ = {res['delta_pp']:+.1f}pp, p = {res['mcnemar']['p_value']:.4f})")

    # Sort results by task index
    task_results.sort(key=lambda x: x["task_idx"])

    # Compile master video
    all_master_frames = []
    for t in task_results:
        if t.get("frames"):
            all_master_frames.extend(t["frames"])
            t["frames"] = None  # Free memory

    master_video_file = os.path.join(VIDEO_DIR, f"maniskill_10tasks_{target_rate}hz_master_comparison.mp4")
    print(f"\n[+] Saving {target_rate} Hz Master Video ({len(all_master_frames)} frames) to {master_video_file}...")
    imageio.mimsave(master_video_file, all_master_frames, fps=target_rate)
    print(f"[SUCCESS] Saved: {master_video_file}")

    total_fold_success = sum(t["folding_success"] for t in task_results)
    total_spline_success = sum(t["spline_success"] for t in task_results)
    total_episodes = sum(t["n_episodes"] for t in task_results)

    agg_fold_rate = total_fold_success / total_episodes
    agg_spline_rate = total_spline_success / total_episodes
    agg_delta_pp = (agg_fold_rate - agg_spline_rate) * 100.0

    return {
        "rate_hz": target_rate,
        "n_episodes_total": total_episodes,
        "aggregate": {
            "fold_success_total": total_fold_success,
            "spline_success_total": total_spline_success,
            "fold_success_rate": agg_fold_rate,
            "spline_success_rate": agg_spline_rate,
            "delta_pp": agg_delta_pp,
        },
        "task_breakdown": task_results,
        "master_video": master_video_file,
    }


def main():
    n_per_task = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    out_file = os.path.join(OUT_DIR, f"maniskill_10tasks_multirate_scoreboard_n{n_per_task}.json")

    res_40hz = run_rate_suite(40, n_per_task=n_per_task, max_workers=workers)
    res_10hz = run_rate_suite(10, n_per_task=n_per_task, max_workers=workers)

    scoreboard = {
        "n_per_task": n_per_task,
        "40hz": res_40hz,
        "10hz": res_10hz,
    }

    Path(out_file).write_text(json.dumps(scoreboard, indent=2))
    print(f"\n[ALL COMPLETE] Full Multi-Rate Scoreboard Saved to {out_file}")


if __name__ == "__main__":
    main()
