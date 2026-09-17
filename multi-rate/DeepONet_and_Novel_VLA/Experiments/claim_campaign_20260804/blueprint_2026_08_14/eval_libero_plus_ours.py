"""LIBERO-Plus Robustness Evaluation: Ours (TAC-Fold) vs. Cubic Spline.

Evaluates trained SmolVLA-DeepONet on LIBERO-Plus across perturbed tasks:
- Compares Ours (TAC-Fold) against Cubic Spline on identical perturbed tasks/seeds.
- Renders synchronized side-by-side comparison video: `libero_plus_ours_vs_spline.mp4`.
- Labels methods explicitly as 'CUBIC SPLINE' and 'OURS'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from collections import deque
import types

import cv2
import imageio
import numpy as np
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator

# Setup paths for Blackwell environment
LIBERO_EXP_DIR = "/home/user/DeepONet_and_Novel_VLA/Experiments/dfof_libero_20260815"
if LIBERO_EXP_DIR not in sys.path:
    sys.path.insert(0, LIBERO_EXP_DIR)

import libero_plus_wrapper as LP
from libero_plus_wrapper import LiberoPlusEnv, list_perturbed_tasks, CATEGORIES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NATIVE_HZ = 20
WINDOW_STEPS = 40
POSE_DIMS = 6
ACT_DIM = 7
SUITE_DATASET = {"libero_spatial": "lerobot/libero_spatial_image"}

TARGET_H, TARGET_W = 256, 256
OUT_DIR = os.path.expanduser("~/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14")
VIDEO_DIR = os.path.join(OUT_DIR, "comparison_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)


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


def ours_tac_resample(native: torch.Tensor, lengths: list[float]) -> torch.Tensor:
    x = np.arange(native.shape[1] + 1, dtype=np.float64)
    q = np.concatenate([[0.0], np.cumsum(lengths)])
    v = native.detach().cpu().double().numpy()
    cum = np.concatenate([np.zeros((v.shape[0], 1, v.shape[2])), np.cumsum(v, axis=1)], axis=1)
    out = np.empty((v.shape[0], len(lengths), v.shape[2]), dtype=np.float64)

    for b in range(v.shape[0]):
        slopes = compute_akima_slopes(x, cum[b])
        resampled_cum = np.empty((len(lengths) + 1, v.shape[2]), dtype=np.float64)
        resampled_cum[0] = cum[b][0]
        for i in range(len(lengths)):
            x0, x1 = q[i], q[i + 1]
            # Find index in x
            idx0 = min(int(np.floor(x0)), len(x) - 2)
            idx1 = idx0 + 1
            resampled_cum[i + 1] = eval_hermite_cubic(
                x[idx0], x[idx1], cum[b][idx0], cum[b][idx1], slopes[idx0], slopes[idx1], np.array([x1])
            )[0]
        out[b] = np.diff(resampled_cum, axis=0) / np.asarray(lengths)[:, None]

    return torch.as_tensor(out, device=native.device, dtype=native.dtype)


def cubic_spline_candidates(native: torch.Tensor, lengths: list[float]) -> torch.Tensor:
    x = np.arange(native.shape[1] + 1, dtype=np.float64)
    q = np.concatenate([[0.0], np.cumsum(lengths)])
    v = native.detach().cpu().double().numpy()
    cum = np.concatenate([np.zeros((v.shape[0], 1, v.shape[2])), np.cumsum(v, axis=1)], axis=1)
    out = np.empty((v.shape[0], len(lengths), v.shape[2]), dtype=np.float64)
    for b in range(v.shape[0]):
        out[b] = np.diff(CubicSpline(x, cum[b], axis=0)(q), axis=0) / np.asarray(lengths)[:, None]
    return torch.as_tensor(out, device=native.device, dtype=native.dtype)


def create_header_banner(width: int, height: int, title: str) -> np.ndarray:
    banner = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (width // 2 - 250, height // 2 + 6),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return banner


def draw_sub_label(frame: np.ndarray, text: str, is_success: bool) -> np.ndarray:
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 230, 0) if is_success else (0, 60, 240)
    tag = "SUCCESS" if is_success else "FAILED"
    full_text = f"{text}: {tag}"
    cv2.rectangle(img, (6, 6), (230, 32), (0, 0, 0), -1)
    cv2.putText(img, full_text, (10, 24), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def decode_chunk_arm(policy, batch, arm: str, rate: int = 10):
    n = rate * WINDOW_STEPS // NATIVE_HZ
    lengths = [WINDOW_STEPS / n] * n
    weights = torch.tensor(lengths, device=DEV, dtype=torch.float32)

    lower = policy.action_min if hasattr(policy, "action_min") else torch.full((ACT_DIM,), -1.0, device=DEV)
    upper = policy.action_max if hasattr(policy, "action_max") else torch.full((ACT_DIM,), 1.0, device=DEV)

    native = policy._get_action_chunk(batch)[:, :WINDOW_STEPS, :]
    native = torch.maximum(torch.minimum(native, upper), lower)
    starts = torch.tensor(np.floor(np.cumsum([0.0] + lengths[:-1])).astype(int), device=DEV)

    if arm == "ours":
        candidate = ours_tac_resample(native, lengths)
    elif arm == "spline":
        candidate = cubic_spline_candidates(native, lengths)
    else:
        raise ValueError(arm)

    pose = candidate[:, :, :POSE_DIMS]
    gripper = native.index_select(1, starts)[:, :, POSE_DIMS:]
    commands = torch.cat([pose, gripper], dim=-1)
    return commands


@torch.no_grad()
def run_rollout_with_video(policy, pre, post, env, task_description, max_steps, seed, arm: str, rate: int = 10):
    from evaluate_plus import plus_obs_to_policy_input
    policy.reset()
    obs = env.reset(seed=seed)
    n_exec = max(1, int(round(0.4 * rate)))
    queue = deque()
    frames = []

    for _ in range(max_steps):
        # Capture rendering
        raw_frame = obs["agentview_image"]
        if raw_frame.shape[:2] != (TARGET_H, TARGET_W):
            raw_frame = cv2.resize(raw_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        frames.append(raw_frame)

        if not queue:
            pin = pre(plus_obs_to_policy_input(obs, task_description))
            pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch = policy._prepare_batch(pin)
                chunk = decode_chunk_arm(policy, batch, arm=arm, rate=rate)
            queue.extend(list(chunk.transpose(0, 1))[:n_exec])

        action = queue.popleft()
        a = post(action).to("cpu").float().numpy().reshape(-1)
        obs, _r, done, _i = env.step(a)
        if env.check_success():
            raw_frame = obs["agentview_image"]
            if raw_frame.shape[:2] != (TARGET_H, TARGET_W):
                raw_frame = cv2.resize(raw_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            frames.append(raw_frame)
            return True, frames
        if done:
            break
    return False, frames


def main():
    ckpt = "/home/user/DeepONet_and_Novel_VLA/Experiments/dfof_libero_20260815/dfof_libero_8300_s0/checkpoints/8300"
    suite = "libero_spatial"
    n_tasks = 20
    max_steps = 250
    rate = 10  # 10 Hz multirate test

    print(f"=======================================================")
    print(f"EVALUATING OURS VS SPLINE ON LIBERO-PLUS ({n_tasks} Tasks, {rate} Hz)")
    print(f"=======================================================")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from modeling_smolvla_deeponet_v2 import SmolVLADeepONetPolicy

    stats = LeRobotDatasetMetadata(SUITE_DATASET[suite]).stats
    policy = SmolVLADeepONetPolicy.from_pretrained(ckpt, deeponet_head="dfof").to(DEV)
    policy.configure_action_stats(
        torch.as_tensor(stats["action"]["mean"], dtype=torch.float32),
        torch.as_tensor(stats["action"]["std"], dtype=torch.float32),
    )
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)

    bench, tasks = list_perturbed_tasks(suite)
    sampled_tasks = tasks[:n_tasks]

    ours_results = []
    spline_results = []
    all_master_frames = []

    for idx, t in enumerate(sampled_tasks, start=1):
        task_name = t.get("name", f"Task_{idx}")
        desc = t.get("language_instruction", "manipulation task")
        print(f"[{idx}/{n_tasks}] Evaluating {t['category']} task: {task_name}...")

        # Arm 1: Cubic Spline
        env_spline = LiberoPlusEnv(bench, t["index"], img_size=TARGET_H, control_freq=rate, scale_pose_deltas=True)
        succ_spline, frames_spline = run_rollout_with_video(
            policy, pre, post, env_spline, desc, max_steps, seed=1000 + idx, arm="spline", rate=rate
        )
        env_spline.close()
        spline_results.append(succ_spline)

        # Arm 2: Ours
        env_ours = LiberoPlusEnv(bench, t["index"], img_size=TARGET_H, control_freq=rate, scale_pose_deltas=True)
        succ_ours, frames_ours = run_rollout_with_video(
            policy, pre, post, env_ours, desc, max_steps, seed=1000 + idx, arm="ours", rate=rate
        )
        env_ours.close()
        ours_results.append(succ_ours)

        print(f"    -> Spline: {'SUCCESS' if succ_spline else 'FAILED'} | Ours: {'SUCCESS' if succ_ours else 'FAILED'}")

        # Render side-by-side video segment
        max_len = max(len(frames_spline), len(frames_ours))
        while len(frames_spline) < max_len:
            frames_spline.append(frames_spline[-1])
        while len(frames_ours) < max_len:
            frames_ours.append(frames_ours[-1])

        banner_title = f"LIBERO-PLUS | {t['category'].upper()}: {task_name[:32]}"
        banner = create_header_banner(TARGET_W * 2, 36, banner_title)

        for f_s, f_o in zip(frames_spline, frames_ours):
            s_ann = draw_sub_label(f_s, "CUBIC SPLINE", succ_spline)
            o_ann = draw_sub_label(f_o, "OURS", succ_ours)
            side_by_side = np.concatenate([s_ann, o_ann], axis=1)
            full_frame = np.concatenate([banner, side_by_side], axis=0)
            all_master_frames.append(full_frame)

        all_master_frames.extend([all_master_frames[-1]] * 10)  # Pause at end of episode

    video_out = os.path.join(VIDEO_DIR, "libero_plus_ours_vs_spline.mp4")
    print(f"\n[+] Saving LIBERO-Plus Master Video ({len(all_master_frames)} frames) to {video_out}...")
    imageio.mimsave(video_out, all_master_frames, fps=20)
    print(f"[SUCCESS] Saved video: {video_out}")

    ours_sr = float(np.mean(ours_results))
    spline_sr = float(np.mean(spline_results))
    delta_pp = (ours_sr - spline_sr) * 100.0

    scoreboard = {
        "n_tasks": n_tasks,
        "rate_hz": rate,
        "ours_success_rate": ours_sr,
        "spline_success_rate": spline_sr,
        "delta_pp": delta_pp,
        "ours_success_count": int(np.sum(ours_results)),
        "spline_success_count": int(np.sum(spline_results)),
        "video_path": video_out,
    }

    out_json = os.path.join(OUT_DIR, "libero_plus_ours_results.json")
    Path(out_json).write_text(json.dumps(scoreboard, indent=2))
    print(f"\n[ALL COMPLETE] LIBERO-Plus Scoreboard Saved to {out_json}")
    print(f"Ours SR: {ours_sr*100:.1f}% | Spline SR: {spline_sr*100:.1f}% (Δ = {delta_pp:+.1f} pp)")


if __name__ == "__main__":
    main()
