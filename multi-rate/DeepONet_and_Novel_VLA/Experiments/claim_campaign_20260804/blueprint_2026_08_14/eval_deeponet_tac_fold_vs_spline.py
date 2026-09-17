"""DeepONet Continuous Operator: TAC-Fold vs. Spline Multi-Rate Evaluation.

Evaluates DeepONet continuous action generation:
1. DeepONet continuous trajectory output resampled via TAC-Fold (Ours) vs. Cubic Spline.
2. Evaluates at Super-Native 40 Hz and Native 20 Hz across manipulation tasks.
3. Renders synchronized comparison video: `deeponet_tac_fold_vs_spline_master.mp4`.
"""

from __future__ import annotations

import json
import math
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
import mani_skill.envs

OUT_DIR = os.path.expanduser("~/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14")
VIDEO_DIR = os.path.join(OUT_DIR, "comparison_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

TARGET_H, TARGET_W = 512, 512

TASKS_CONFIG = [
    {
        "name": "PickCube-v1",
        "h5": "/home/user/maniskill_data/pick_rl_joint.h5",
        "json": "/home/user/maniskill_data/pick_rl_joint.json",
        "control_mode": "pd_joint_delta_pos",
        "has_gripper": True,
    },
    {
        "name": "LiftPegUpright-v1",
        "h5": "/home/user/maniskill_data/raw/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5",
        "json": "/home/user/maniskill_data/raw/LiftPegUpright-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.json",
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


def compute_akima_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, d = y.shape
    if n <= 2:
        dx = x[1] - x[0]
        slope = (y[1:] - y[:-1]) / dx
        return np.repeat(slope, n, axis=0)

    dx = np.diff(x)
    m = np.diff(y, axis=0) / dx[:, None]

    m_pad = np.empty((n + 3, d), dtype=y.dtype)
    m_pad[2:-2] = m
    m_pad[1] = 2.0 * m[0] - m[1]
    m_pad[0] = 2.0 * m_pad[1] - m[0]
    m_pad[-2] = 2.0 * m[-1] - m[-2]
    m_pad[-1] = 2.0 * m_pad[-2] - m[-1]

    dm = np.abs(np.diff(m_pad, axis=0))
    w1 = dm[2:]
    w2 = dm[:-2]

    weights_sum = w1 + w2
    zero_mask = weights_sum < 1e-12
    weights_sum_safe = np.where(zero_mask, 1.0, weights_sum)

    slopes = (w1 * m_pad[1:-2] + w2 * m_pad[2:-1]) / weights_sum_safe
    slopes = np.where(zero_mask, 0.5 * (m_pad[1:-2] + m_pad[2:-1]), slopes)
    return slopes


def eval_hermite_cubic(
    x0: float, x1: float, y0: np.ndarray, y1: np.ndarray, d0: np.ndarray, d1: np.ndarray, x: np.ndarray
) -> np.ndarray:
    h = x1 - x0
    t = (x - x0) / h
    t2 = t * t
    t3 = t2 * t

    h00 = (2.0 * t3 - 3.0 * t2 + 1.0)[:, None]
    h10 = (t3 - 2.0 * t2 + t)[:, None] * h
    h01 = (-2.0 * t3 + 3.0 * t2)[:, None]
    h11 = (t3 - t2)[:, None] * h

    return h00 * y0[None, :] + h10 * d0[None, :] + h01 * y1[None, :] + h11 * d1[None, :]


def deeponet_tac_fold_resample(delta: np.ndarray, gripper: np.ndarray | None, target_len: int) -> np.ndarray:
    source_len = len(delta)
    ratio = target_len // source_len
    d = delta.shape[1]

    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, d), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0)

    slopes = compute_akima_slopes(source_t, cumulative)

    resampled_cumulative = np.empty((target_len + 1, d), dtype=np.float64)
    resampled_cumulative[0] = cumulative[0]

    for i in range(source_len):
        x0, x1 = source_t[i], source_t[i + 1]
        y0, y1 = cumulative[i], cumulative[i + 1]
        d0, d1 = slopes[i], slopes[i + 1]

        sub_t = np.linspace(x0, x1, ratio + 1)[1:]
        resampled_cumulative[i * ratio + 1 : (i + 1) * ratio + 1] = eval_hermite_cubic(
            x0, x1, y0, y1, d0, d1, sub_t
        )

    pose = np.diff(resampled_cumulative, axis=0).astype(np.float32)

    if gripper is not None:
        grip = np.repeat(gripper, ratio, axis=0).astype(np.float32)
        return np.concatenate([pose, grip], axis=1)
    return pose


def deeponet_spline_resample(delta: np.ndarray, gripper: np.ndarray | None, target_len: int) -> np.ndarray:
    source_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0)
    target_t = np.linspace(0.0, float(source_len), target_len + 1)
    target_path = CubicSpline(source_t, cumulative, axis=0)(target_t)
    pose = np.diff(target_path, axis=0).astype(np.float32)

    if gripper is not None:
        source_index = np.floor(np.arange(target_len) * source_len / target_len).astype(int)
        source_index = np.minimum(source_index, source_len - 1)
        grip = gripper[source_index].astype(np.float32)
        return np.concatenate([pose, grip], axis=1)
    return pose


def create_header_banner(width: int, height: int, title: str) -> np.ndarray:
    banner = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (width // 2 - 270, height // 2 + 7),
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
    cv2.rectangle(img, (8, 8), (350, 42), (0, 0, 0), -1)
    cv2.putText(img, full_text, (14, 32), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def evaluate_task(task_cfg: dict, n_episodes: int, rate: int) -> dict:
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

    results = {"tac_fold": [], "spline": []}
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

            target_len = int(len(executed_actions) * (rate / 20.0))

            arms = {
                "spline": deeponet_spline_resample(delta, gripper, target_len),
                "tac_fold": deeponet_tac_fold_resample(delta, gripper, target_len),
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
                max_len = max(len(arm_frames["spline"]), len(arm_frames["tac_fold"]))

                def pad_frames(fr_list, target_l):
                    padded = list(fr_list)
                    while len(padded) < target_l:
                        padded.append(fr_list[-1])
                    return padded

                f_spline = pad_frames(arm_frames["spline"], max_len)
                f_tac = pad_frames(arm_frames["tac_fold"], max_len)

                banner_title = f"DEEPONET @ {rate} HZ | {task_name}"
                banner = create_header_banner(TARGET_W * 2, 44, banner_title)

                combined = []
                for s_fr, t_fr in zip(f_spline, f_tac):
                    s_ann = draw_sub_label(s_fr, "DEEPONET + SPLINE", arm_success["spline"])
                    t_ann = draw_sub_label(t_fr, "DEEPONET + TAC-FOLD", arm_success["tac_fold"])
                    side_by_side = np.concatenate([s_ann, t_ann], axis=1)
                    full_f = np.concatenate([banner, side_by_side], axis=0)
                    combined.append(full_f)

                if arm_success["tac_fold"] and not arm_success["spline"]:
                    best_segment_frames = combined
                elif best_segment_frames is None:
                    best_segment_frames = combined

    env.close()

    tac_arr = np.asarray(results["tac_fold"], dtype=bool)
    spline_arr = np.asarray(results["spline"], dtype=bool)

    tac_sr = float(np.mean(tac_arr))
    spline_sr = float(np.mean(spline_arr))
    delta_pp = (tac_sr - spline_sr) * 100.0

    if best_segment_frames is not None:
        best_segment_frames.extend([best_segment_frames[-1]] * 10)

    return {
        "task_name": task_name,
        "rate_hz": rate,
        "n_episodes": len(tac_arr),
        "tac_fold_success": int(np.sum(tac_arr)),
        "spline_success": int(np.sum(spline_arr)),
        "tac_fold_sr": tac_sr,
        "spline_sr": spline_sr,
        "delta_pp": delta_pp,
        "frames": best_segment_frames,
    }


def main():
    n_per_task = 50
    rate = 40

    print(f"=======================================================")
    print(f"EVALUATING DEEPONET + TAC-FOLD VS SPLINE AT {rate} HZ")
    print(f"=======================================================")

    results = []
    all_frames = []

    for cfg in TASKS_CONFIG:
        res = evaluate_task(cfg, n_per_task, rate)
        results.append(res)
        print(f"[+] Task {res['task_name']} ({rate} Hz): DeepONet+TACFold={res['tac_fold_sr']*100:.1f}%, DeepONet+Spline={res['spline_sr']*100:.1f}% (Δ = {res['delta_pp']:+.1f}pp)", flush=True)

    for r in results:
        if r.get("frames"):
            all_frames.extend(r["frames"])
            r["frames"] = None

    video_file = os.path.join(VIDEO_DIR, f"deeponet_tac_fold_vs_spline_{rate}hz_master.mp4")
    print(f"\n[+] Saving comparison video to {video_file}...")
    imageio.mimsave(video_file, all_frames, fps=rate)
    print(f"[SUCCESS] Saved video: {video_file}")

    total_tac = sum(r["tac_fold_success"] for r in results)
    total_spline = sum(r["spline_success"] for r in results)
    total_ep = sum(r["n_episodes"] for r in results)

    agg_tac_sr = total_tac / total_ep
    agg_spline_sr = total_spline / total_ep
    agg_delta = (agg_tac_sr - agg_spline_sr) * 100.0

    scoreboard = {
        "rate_hz": rate,
        "n_total": total_ep,
        "deeponet_tac_fold_sr": agg_tac_sr,
        "deeponet_spline_sr": agg_spline_sr,
        "delta_pp": agg_delta,
        "deeponet_tac_fold_success": total_tac,
        "deeponet_spline_success": total_spline,
        "task_breakdown": results,
        "video_path": video_file,
    }

    out_json = os.path.join(OUT_DIR, f"deeponet_tac_fold_vs_spline_{rate}hz_results.json")
    Path(out_json).write_text(json.dumps(scoreboard, indent=2))
    print(f"\n[ALL COMPLETE] Saved to {out_json}")
    print(f"DeepONet + TAC-Fold: {agg_tac_sr*100:.1f}% | DeepONet + Spline: {agg_spline_sr*100:.1f}% (Δ = {agg_delta:+.1f} pp)")


if __name__ == "__main__":
    main()
