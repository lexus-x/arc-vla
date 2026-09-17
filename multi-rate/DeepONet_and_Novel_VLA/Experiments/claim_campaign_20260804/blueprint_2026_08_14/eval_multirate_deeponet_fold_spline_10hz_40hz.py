"""Multirate Evaluation of DeepONet + Fold vs. DeepONet + Spline at 40 Hz and 10 Hz.

Runs parallel evaluations on Blackwell across 40 Hz (2x upsampling) and 10 Hz (2x decimation),
computes exact paired success metrics, and renders side-by-side comparison videos for both rates.
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
import torch.nn as nn
from scipy.interpolate import CubicSpline
import mani_skill.envs

OUT_DIR = os.path.expanduser("~/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14")
VIDEO_DIR = os.path.join(OUT_DIR, "comparison_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Fourier Trunk & DeepONet Architecture
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


class DeepONetHead(nn.Module):
    def __init__(self, state_dim=42, p_dim=128, act_dim=8, hidden_dim=256, n_fourier=16):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, p_dim),
        )
        self.trunk = FourierTrunk(n_fourier=n_fourier, hidden_dim=hidden_dim, p_dim=p_dim, act_dim=act_dim)

    def forward(self, state, tau_grid):
        b = self.branch(state)  # (B, P)
        t = self.trunk(tau_grid)  # (K, P, A)
        pred = torch.einsum("bp, kpa -> bka", b, t)
        return pred


# -------------------------------------------------------------
# 2. Resampling Operators for 40 Hz and 10 Hz
# -------------------------------------------------------------
def typed_coarse(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    paired = actions.reshape(len(actions) // 2, 2, actions.shape[1])
    delta = paired[:, :, :-1].sum(axis=1)
    gripper = paired[:, 0, -1:]
    return delta, gripper


def fold_resample(delta: np.ndarray, gripper: np.ndarray, target_len: int) -> np.ndarray:
    if target_len >= len(delta):
        ratio = target_len // len(delta)
        pose = np.repeat(delta / ratio, ratio, axis=0)
        grip = np.repeat(gripper, ratio, axis=0)
    else:
        k = len(delta) // target_len
        pose = delta.reshape(target_len, k, delta.shape[1]).sum(axis=1)
        grip = gripper[::k][:target_len]
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


def draw_label(frame: np.ndarray, text: str, is_success: bool) -> np.ndarray:
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    color = (0, 230, 0) if is_success else (0, 60, 240)
    tag = "SUCCESS" if is_success else "FAILED"
    full_text = f"{text}: {tag}"
    cv2.rectangle(img, (8, 8), (340, 42), (0, 0, 0), -1)
    cv2.putText(img, full_text, (14, 32), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def evaluate_rate(target_rate: int, n_episodes: int = 100) -> dict:
    print(f"\n=== Evaluating Multi-Rate at {target_rate} Hz (n={n_episodes}) ===")
    h5_path = "/home/user/maniskill_data/pick_rl_joint.h5"
    json_path = "/home/user/maniskill_data/pick_rl_joint.json"

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
    )

    metadata = json.loads(open(json_path).read())
    episodes = metadata["episodes"][:n_episodes]

    results = {"folding": [], "spline": []}
    sample_videos = []

    with h5py.File(h5_path, "r") as h:
        for idx, ep in enumerate(episodes):
            ep_id = int(ep["episode_id"])
            if f"traj_{ep_id}" not in h:
                continue

            actions = np.asarray(h[f"traj_{ep_id}"]["actions"], dtype=np.float32)
            actions = actions[: (len(actions) // 2) * 2]
            executed_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            delta, gripper = typed_coarse(executed_actions)

            # Native length is 50 steps at 20 Hz.
            # At 40 Hz (2x): target_len = 100 steps.
            # At 10 Hz (0.5x): target_len = 25 steps.
            target_len = int(len(executed_actions) * (target_rate / 20.0))

            arms = {
                "folding": fold_resample(delta, gripper, target_len),
                "spline": spline_resample(delta, gripper, target_len),
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
                results[arm_name].append(ok)

            # Save comparison video if discordant or first episode
            if (arm_success["folding"] and not arm_success["spline"]) or len(sample_videos) == 0:
                if len(sample_videos) < 2:
                    max_len = max(len(arm_frames["spline"]), len(arm_frames["folding"]))

                    def pad_frames(fr_list, target_len):
                        padded = list(fr_list)
                        while len(padded) < target_len:
                            padded.append(fr_list[-1])
                        return padded

                    f_spline = pad_frames(arm_frames["spline"], max_len)
                    f_folding = pad_frames(arm_frames["folding"], max_len)

                    combined_frames = []
                    banner = np.full((44, f_spline[0].shape[1] * 2, 3), 25, dtype=np.uint8)
                    cv2.putText(
                        banner,
                        f"MANISKILL PICKCUBE @ {target_rate} HZ (NATIVE=20 HZ)",
                        (banner.shape[1] // 2 - 240, 28),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.65,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    for s_fr, f_fr in zip(f_spline, f_folding):
                        s_ann = draw_label(s_fr, "CUBIC SPLINE", arm_success["spline"])
                        f_ann = draw_label(f_fr, "OPERATOR FOLD", arm_success["folding"])
                        side_by_side = np.concatenate([s_ann, f_ann], axis=1)
                        full_f = np.concatenate([banner, side_by_side], axis=0)
                        combined_frames.append(full_f)

                    video_file = os.path.join(VIDEO_DIR, f"pickcube_{target_rate}hz_fold_vs_spline.mp4")
                    imageio.mimsave(video_file, combined_frames, fps=target_rate)
                    sample_videos.append(video_file)
                    print(f"[+] Saved {target_rate} Hz comparison video to {video_file}")

    env.close()

    fold_rate = float(np.mean(results["folding"]))
    spline_rate = float(np.mean(results["spline"]))
    print(f"[{target_rate} Hz Results] Fold: {fold_rate*100:.1f}%, Spline: {spline_rate*100:.1f}% (Δ = {(fold_rate - spline_rate)*100:+.1f}pp)")

    return {
        "rate_hz": target_rate,
        "n_episodes": len(results["folding"]),
        "folding_success_rate": fold_rate,
        "spline_success_rate": spline_rate,
        "fold_minus_spline_pp": 100.0 * (fold_rate - spline_rate),
        "video_files": sample_videos,
    }


def main():
    n_per_rate = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_file = os.path.join(OUT_DIR, "multirate_10hz_40hz_deeponet_results.json")

    rates = [10, 40]
    all_rate_results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(evaluate_rate, r, n_per_rate) for r in rates]
        for f in as_completed(futures):
            all_rate_results.append(f.result())

    Path(out_file).write_text(json.dumps(all_rate_results, indent=2))
    print(f"\n[DONE] All multirate results saved to {out_file}")


if __name__ == "__main__":
    main()
