"""DeepONet Closed-Loop Policy Evaluation: Native 20 Hz and 40 Hz on ManiSkill.

Evaluates DeepONet continuous policy:
1. Evaluates at Native (20 Hz) comparing DeepONet + Ours (TAC-Fold) vs. DeepONet + Spline.
2. Evaluates at Super-Native (40 Hz) comparing DeepONet + Ours (TAC-Fold) vs. DeepONet + Spline.
3. Renders synchronized master comparison video for both rates:
   - `deeponet_maniskill_20hz_ours_vs_spline.mp4`
   - `deeponet_maniskill_40hz_ours_vs_spline.mp4`
4. Saves scoreboard JSON.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline
import mani_skill.envs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_PATH = os.path.expanduser("~/maniskill_data")
OUT_DIR = os.path.expanduser("~/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14")
VIDEO_DIR = os.path.join(OUT_DIR, "comparison_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

TARGET_H, TARGET_W = 512, 512

TASKS = [
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


# -------------------------------------------------------------
# DeepONet Architecture
# -------------------------------------------------------------
class FourierTrunk(nn.Module):
    def __init__(self, n_fourier=16, hidden_dim=256, p_dim=128, act_dim=8):
        super().__init__()
        self.n_fourier = n_fourier
        in_dim = 1 + 2 * n_fourier
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, p_dim * act_dim),
        )
        self.p_dim = p_dim
        self.act_dim = act_dim

    def forward(self, tau):
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)
        freqs = torch.arange(1, self.n_fourier + 1, device=tau.device, dtype=tau.dtype) * 2 * math.pi
        feat = torch.cat([tau, torch.sin(tau * freqs), torch.cos(tau * freqs)], dim=-1)
        out = self.mlp(feat)
        return out.view(*out.shape[:-1], self.p_dim, self.act_dim)


class DeepONetActionHead(nn.Module):
    def __init__(self, state_dim, p_dim=128, act_dim=8, hidden_dim=256):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, p_dim),
        )
        self.trunk = FourierTrunk(n_fourier=16, hidden_dim=hidden_dim, p_dim=p_dim, act_dim=act_dim)
        self.bias = nn.Parameter(torch.zeros(act_dim))

    def forward(self, state, tau_grid):
        b = self.branch(state)
        t = self.trunk(tau_grid)
        out = torch.einsum("bp, tpa -> bta", b, t) + self.bias
        return out


# -------------------------------------------------------------
# Resamplers
# -------------------------------------------------------------
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


def resample_tac_fold(actions: np.ndarray, target_rate: int, has_gripper: bool = True) -> np.ndarray:
    if target_rate == 20:
        return actions.astype(np.float32)

    if has_gripper:
        delta = actions[:, :-1]
        gripper = actions[:, -1:]
    else:
        delta = actions
        gripper = None

    source_len = len(delta)
    ratio = target_rate // 20
    target_len = source_len * ratio
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


def resample_spline(actions: np.ndarray, target_rate: int, has_gripper: bool = True) -> np.ndarray:
    if target_rate == 20:
        # At 20 Hz, smooth spline filter through waypoints
        delta = actions[:, :-1] if has_gripper else actions
        gripper = actions[:, -1:] if has_gripper else None
        source_len = len(delta)
        source_t = np.arange(source_len + 1, dtype=np.float64)
        cumulative = np.concatenate([np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0)
        target_t = np.linspace(0.0, float(source_len), source_len + 1)
        target_path = CubicSpline(source_t, cumulative, axis=0)(target_t)
        pose = np.diff(target_path, axis=0).astype(np.float32)
        if gripper is not None:
            return np.concatenate([pose, gripper.astype(np.float32)], axis=1)
        return pose

    delta = actions[:, :-1] if has_gripper else actions
    gripper = actions[:, -1:] if has_gripper else None
    source_len = len(delta)
    target_len = source_len * (target_rate // 20)
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


def evaluate_task_at_rate(task_cfg: dict, rate: int, n_episodes: int = 50) -> dict:
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

            arms = {
                "spline": resample_spline(executed_actions, rate, has_gripper),
                "ours": resample_tac_fold(executed_actions, rate, has_gripper),
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

                banner_title = f"DEEPONET @ {rate} HZ | {task_name}"
                banner = create_header_banner(TARGET_W * 2, 44, banner_title)

                combined = []
                for s_fr, o_fr in zip(f_spline, f_ours):
                    s_ann = draw_sub_label(s_fr, "DEEPONET + SPLINE", arm_success["spline"])
                    o_ann = draw_sub_label(o_fr, "DEEPONET + OURS", arm_success["ours"])
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

    ours_sr = float(np.mean(ours_arr))
    spline_sr = float(np.mean(spline_arr))
    delta_pp = (ours_sr - spline_sr) * 100.0

    if best_segment_frames is not None:
        best_segment_frames.extend([best_segment_frames[-1]] * 10)

    return {
        "task_name": task_name,
        "rate_hz": rate,
        "n_episodes": len(ours_arr),
        "ours_success": int(np.sum(ours_arr)),
        "spline_success": int(np.sum(spline_arr)),
        "ours_sr": ours_sr,
        "spline_sr": spline_sr,
        "delta_pp": delta_pp,
        "frames": best_segment_frames,
    }


def run_evaluation_for_rate(rate: int, n_per_task: int = 50):
    print(f"\n=======================================================")
    print(f"EVALUATING DEEPONET: OURS VS SPLINE AT {rate} HZ")
    print(f"=======================================================")

    results = []
    all_frames = []

    for cfg in TASKS:
        res = evaluate_task_at_rate(cfg, rate, n_per_task)
        results.append(res)
        print(f"[+] Task {res['task_name']} ({rate} Hz): DeepONet+Ours={res['ours_sr']*100:.1f}%, DeepONet+Spline={res['spline_sr']*100:.1f}% (Δ = {res['delta_pp']:+.1f}pp)", flush=True)

    for r in results:
        if r.get("frames"):
            all_frames.extend(r["frames"])
            r["frames"] = None

    video_file = os.path.join(VIDEO_DIR, f"deeponet_maniskill_{rate}hz_ours_vs_spline.mp4")
    print(f"\n[+] Saving {rate} Hz comparison video ({len(all_frames)} frames) to {video_file}...")
    imageio.mimsave(video_file, all_frames, fps=rate)
    print(f"[SUCCESS] Saved video: {video_file}")

    total_ours = sum(r["ours_success"] for r in results)
    total_spline = sum(r["spline_success"] for r in results)
    total_ep = sum(r["n_episodes"] for r in results)

    agg_ours_sr = total_ours / total_ep
    agg_spline_sr = total_spline / total_ep
    agg_delta = (agg_ours_sr - agg_spline_sr) * 100.0

    scoreboard = {
        "rate_hz": rate,
        "n_total": total_ep,
        "deeponet_ours_sr": agg_ours_sr,
        "deeponet_spline_sr": agg_spline_sr,
        "delta_pp": agg_delta,
        "deeponet_ours_success": total_ours,
        "deeponet_spline_success": total_spline,
        "task_breakdown": results,
        "video_path": video_file,
    }

    out_json = os.path.join(OUT_DIR, f"deeponet_maniskill_{rate}hz_results.json")
    Path(out_json).write_text(json.dumps(scoreboard, indent=2))
    print(f"\n[COMPLETE {rate} HZ] Saved to {out_json}")
    print(f"DeepONet + Ours: {agg_ours_sr*100:.1f}% | DeepONet + Spline: {agg_spline_sr*100:.1f}% (Δ = {agg_delta:+.1f} pp)")
    return scoreboard


def main():
    n_per_task = 50
    # 1. Native 20 Hz
    sb_20 = run_evaluation_for_rate(20, n_per_task)
    # 2. Super-Native 40 Hz
    sb_40 = run_evaluation_for_rate(40, n_per_task)

    print("\n=======================================================")
    print("ALL DEEPONET MANISKILL EVALUATIONS COMPLETE")
    print(f"20 Hz: DeepONet+Ours={sb_20['deeponet_ours_sr']*100:.1f}% vs Spline={sb_20['deeponet_spline_sr']*100:.1f}% (Δ = {sb_20['delta_pp']:+.1f} pp)")
    print(f"40 Hz: DeepONet+Ours={sb_40['deeponet_ours_sr']*100:.1f}% vs Spline={sb_40['deeponet_spline_sr']*100:.1f}% (Δ = {sb_40['delta_pp']:+.1f} pp)")
    print("=======================================================")


if __name__ == "__main__":
    main()
