#!/usr/bin/env python
"""Paired multi-rate MetaWorld evaluation for a frozen DeepONet checkpoint.

The simulator remains at its native 80 Hz.  Each decoder emits a lower-rate
command which is held for an integer number of native steps.  The 30 Hz arm
uses a deterministic 3/3/2 native-step schedule.  Pose channels share the same
box-and-integral projection; the gripper is always causal sample-and-hold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from safetensors.torch import load_file

from lerobot.envs.configs import MetaworldEnv as MetaworldEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.policies.utils import populate_queues
from lerobot.scripts.lerobot_eval import eval_policy


NATIVE_HZ = 80
WINDOW_STEPS = 32  # fixed 0.4 s physical replan window
POSE_DIMS = 3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hold_lengths(rate: int) -> list[float]:
    """Partition 32 native steps into exactly 0.4*rate command cells."""
    n_commands = rate * WINDOW_STEPS // NATIVE_HZ
    if n_commands * NATIVE_HZ != rate * WINDOW_STEPS:
        raise ValueError(f"rate {rate} does not yield an integer command count")
    if rate == 30:
        # 13/13/14 physics frames at 400 Hz: exact 30 Hz over every 0.1 s.
        lengths = [2.6, 2.6, 2.8] * 4
    else:
        lengths = [WINDOW_STEPS / n_commands] * n_commands
    assert len(lengths) == n_commands and sum(lengths) == WINDOW_STEPS
    assert min(lengths) > 0 and abs(sum(lengths) - WINDOW_STEPS) < 1e-9
    return lengths


def weighted_box_sum_projection(
    candidate: torch.Tensor,
    weights: torch.Tensor,
    target_sum: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    iterations: int = 60,
) -> torch.Tensor:
    """Project (B,N,D) onto bounds and weighted sums, independently per B,D."""
    if candidate.ndim != 3:
        raise ValueError("candidate must have shape (B,N,D)")
    w = weights.view(1, -1, 1).to(candidate)
    lo = lower.view(1, 1, -1).to(candidate)
    hi = upper.view(1, 1, -1).to(candidate)
    target = target_sum.unsqueeze(1).to(candidate)
    # u(lambda)=clip(v-lambda, lo, hi); weighted sum decreases in lambda.
    left = (candidate - hi).amin(dim=1, keepdim=True) - 1.0
    right = (candidate - lo).amax(dim=1, keepdim=True) + 1.0
    for _ in range(iterations):
        mid = (left + right) / 2
        value = (torch.clamp(candidate - mid, lo, hi) * w).sum(dim=1, keepdim=True)
        left = torch.where(value > target, mid, left)
        right = torch.where(value > target, right, mid)
    return torch.clamp(candidate - (left + right) / 2, lo, hi)


def command_centers(lengths: list[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    starts = np.cumsum([0] + lengths[:-1], dtype=float)
    centers = starts + (np.asarray(lengths, dtype=float) - 1.0) / 2.0
    # The trained 50-point grid is indexed 0..49 at the native 80 Hz cadence.
    return torch.tensor(centers / 49.0, device=device, dtype=dtype).unsqueeze(-1)


def pchip_candidates(native: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x = np.arange(native.shape[1], dtype=np.float64)
    q = centers.detach().cpu().numpy().reshape(-1) * 49.0
    values = native.detach().cpu().double().numpy()
    out = np.empty((values.shape[0], len(q), values.shape[2]), dtype=np.float64)
    for batch_index in range(values.shape[0]):
        out[batch_index] = PchipInterpolator(x, values[batch_index], axis=0)(q)
    return torch.as_tensor(out, device=native.device, dtype=native.dtype)


def start_indices(lengths: list[float], device: torch.device) -> torch.Tensor:
    starts = np.floor(np.cumsum([0.0] + lengths[:-1])).astype(int)
    return torch.tensor(starts, device=device, dtype=torch.long)


def decode_chunk(
    policy,
    batch: dict[str, torch.Tensor],
    arm: str,
    rate: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    lengths = hold_lengths(rate)
    weights = torch.tensor(lengths, device=lower.device, dtype=lower.dtype)
    native = policy._get_action_chunk(batch)[:, :WINDOW_STEPS, :]
    native = torch.maximum(torch.minimum(native, upper), lower)
    centers = command_centers(lengths, native.device, native.dtype)
    starts = start_indices(lengths, native.device)

    if arm == "ratefold":
        head = policy.model.deeponet
        original_tau = head.tau
        try:
            head.tau = centers
            candidate = policy._get_action_chunk(batch)[:, :, :]
        finally:
            head.tau = original_tau
    elif arm == "zoh":
        candidate = native.index_select(1, starts)
    elif arm == "pchip":
        candidate = pchip_candidates(native, centers)
    else:
        raise ValueError(f"unknown arm: {arm}")

    candidate = candidate[:, :, : native.shape[-1]]
    target = native[:, :, :POSE_DIMS].sum(dim=1)
    projected_pose = weighted_box_sum_projection(
        candidate[:, :, :POSE_DIMS], weights, target, lower[:POSE_DIMS], upper[:POSE_DIMS]
    )
    # Gripper is typed as an event/hold channel for every arm.
    gripper = native.index_select(1, starts)[:, :, POSE_DIMS:]
    commands = torch.cat([projected_pose, gripper], dim=-1)
    endpoint_error = (
        (projected_pose * weights.view(1, -1, 1)).sum(dim=1) - target
    ).abs().amax()
    saturation = ((projected_pose <= lower[:POSE_DIMS] + 1e-6) |
                  (projected_pose >= upper[:POSE_DIMS] - 1e-6)).float().mean()
    if float(endpoint_error) > 2e-4:
        raise RuntimeError(f"projection endpoint error {float(endpoint_error):.3g}")
    return commands, {
        "max_endpoint_error": float(endpoint_error),
        "pose_saturation_fraction": float(saturation),
    }


def install_decoder(policy, arm: str, rate: int, lower: torch.Tensor, upper: torch.Tensor, telemetry: dict):
    """Patch only action queue construction; retain the checkpoint and LeRobot API."""
    def reset(self):
        type(self).reset(self)
        self._ratefold_queue = deque()

    @torch.no_grad()
    def select_action(self, batch, noise=None, **kwargs):
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=["action"])
        if not getattr(self, "_ratefold_queue", None):
            chunk, stats = decode_chunk(self, batch, arm, rate, lower, upper)
            self._ratefold_queue = deque(chunk.transpose(0, 1))
            telemetry["windows"] += 1
            telemetry["max_endpoint_error"] = max(telemetry["max_endpoint_error"], stats["max_endpoint_error"])
            telemetry["pose_saturation_sum"] += stats["pose_saturation_fraction"]
        return self._ratefold_queue.popleft()

    policy.reset = types.MethodType(reset, policy)
    policy.select_action = types.MethodType(select_action, policy)


def install_env_rate(env, rate: int, horizon_seconds: int = 6) -> None:
    """Run the actual MetaWorld control loop at rate Hz, retaining 400 Hz physics."""
    pattern = hold_lengths(rate)
    native_reset = getattr(env, "_ratefold_native_reset", env.reset)
    native_step = getattr(env, "_ratefold_native_step", env.step)
    env._ratefold_native_reset = native_reset
    env._ratefold_native_step = native_step
    for wrapped in env.envs:
        wrapped._max_episode_steps = horizon_seconds * rate
        wrapped._env.max_path_length = horizon_seconds * rate
        if not hasattr(wrapped, "_ratefold_native_reset"):
            wrapped._ratefold_native_reset = wrapped.reset

            def paired_reset(self, *, seed=None, options=None):
                # MetaWorld v3 otherwise ignores reset(seed) for task randomization.
                self._env.seeded_rand_vec = True
                if seed is not None:
                    self._env.seed(seed)
                return self._ratefold_native_reset(seed=seed, options=options)

            wrapped.reset = types.MethodType(paired_reset, wrapped)

    def reset(self, *args, **kwargs):
        self._ratefold_step_index = 0
        for wrapped in self.envs:
            # Reset settling must always use the native training dynamics.
            wrapped._env.frame_skip = 5
            wrapped._env.action_scale = 0.01
        result = self._ratefold_native_reset(*args, **kwargs)
        hashes = []
        for wrapped in self.envs:
            raw = wrapped._env
            arrays = [raw.data.qpos, raw.data.qvel, raw.data.mocap_pos]
            if hasattr(raw, "_target_pos"):
                arrays.append(np.asarray(raw._target_pos))
            payload = b"".join(np.asarray(value, dtype=np.float64).tobytes() for value in arrays)
            hashes.append(hashlib.sha256(payload).hexdigest())
        self._ratefold_initial_hashes = hashes
        return result

    def step(self, actions):
        frames = round(pattern[self._ratefold_step_index % len(pattern)] * 5)
        for wrapped in self.envs:
            wrapped._env.frame_skip = frames
            # Native max xyz speed is 0.01 m per 5 physics frames.
            wrapped._env.action_scale = 0.01 * frames / 5.0
        self._ratefold_step_index += 1
        return self._ratefold_native_step(actions)

    env.reset = types.MethodType(reset, env)
    env.step = types.MethodType(step, env)


class AffineActionPostprocessor:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.std.to(action) + self.mean.to(action)


def successes(info: dict, fallback_n: int) -> tuple[int, int, list[bool]]:
    episodes = info.get("per_episode", [])
    if episodes:
        flags = []
        for episode in episodes:
            value = episode.get("success", episode.get("is_success", episode.get("pc_success", False)))
            flags.append(bool(value[-1]) if isinstance(value, (list, tuple)) else bool(value))
        return sum(flags), len(flags), flags
    pct = float(info["aggregated"]["pc_success"])
    count = round(pct * fallback_n / 100)
    return count, fallback_n, []


def build(ckpt: str, task: str, batch_size: int, device: str, mean: torch.Tensor, std: torch.Tensor):
    from modeling_smolvla_deeponet_v2 import SmolVLADeepONetPolicy

    cfg = MetaworldEnvConfig(task=task, obs_type="pixels_agent_pos")
    env = make_env(cfg, n_envs=batch_size, use_async_envs=False)[task][0]
    policy = SmolVLADeepONetPolicy.from_pretrained(
        ckpt, ph_enabled=False, deeponet_p=256, deeponet_blocks=3,
        deeponet_queries=8, deeponet_fourier=16, deeponet_head="deeponet",
    ).to(device).eval()
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=ckpt,
        config_filename="policy_preprocessor.json",
        overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = AffineActionPostprocessor(mean, std)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
    return env, policy, pre, post, env_pre, env_post


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--rates", default="5,10,30,40")
    parser.add_argument("--arms", default="ratefold,zoh,pchip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    rates = [int(item) for item in args.rates.split(",") if item.strip()]
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    stats_path = Path(args.ckpt) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    stats = load_file(str(stats_path))
    mean = stats["action.mean"].to(args.device)
    std = stats["action.std"].to(args.device)
    lower = (-torch.ones_like(mean) - mean) / std
    upper = (torch.ones_like(mean) - mean) / std

    fresh_output = {
        "status": "INCOMPLETE",
        "config": vars(args),
        "provenance": {
            "evaluator_sha256": file_sha256(Path(__file__)),
            "checkpoint_model_sha256": file_sha256(Path(args.ckpt) / "model.safetensors"),
        },
        "protocol": {
            "physics_hz": 400,
            "training_control_hz": NATIVE_HZ,
            "window_steps": WINDOW_STEPS,
            "window_seconds": WINDOW_STEPS / NATIVE_HZ,
            "episode_horizon_seconds": 6,
            "pose_dims": POSE_DIMS,
            "shared_projection": "normalized box equivalent to physical [-1,1], weighted native-prefix sum",
            "gripper": "causal ZOH for every arm",
        },
        "cells": {},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists():
        output = json.loads(out.read_text())
        output["status"] = "INCOMPLETE"
    else:
        output = fresh_output

    for task in tasks:
        env, policy, pre, post, env_pre, env_post = build(
            args.ckpt, task, args.batch_size, args.device, mean, std
        )
        existing_task_cells = [
            cell for cell in output["cells"].values() if cell["task"] == task
        ]
        paired_hashes = (
            existing_task_cells[0]["initial_state_hashes"] if existing_task_cells else None
        )
        for rate in rates:
            install_env_rate(env, rate)
            for arm in arms:
                key = f"{task}|{rate}|{arm}"
                if key in output["cells"]:
                    print(key, "SKIP completed", flush=True)
                    continue
                telemetry = {"windows": 0, "max_endpoint_error": 0.0, "pose_saturation_sum": 0.0}
                install_decoder(policy, arm, rate, lower, upper, telemetry)
                info = eval_policy(
                    env=env, policy=policy, env_preprocessor=env_pre,
                    env_postprocessor=env_post, preprocessor=pre, postprocessor=post,
                    n_episodes=args.episodes, max_episodes_rendered=0,
                    videos_dir=None, start_seed=args.start_seed,
                )
                k, n, flags = successes(info, args.episodes)
                initial_hashes = list(env._ratefold_initial_hashes)
                if paired_hashes is None:
                    paired_hashes = initial_hashes
                elif initial_hashes != paired_hashes:
                    raise RuntimeError(f"paired initial-state mismatch for {task}|{rate}|{arm}")
                output["cells"][key] = {
                    "task": task, "rate_hz": rate, "arm": arm,
                    "successes": k, "n": n, "success_flags": flags,
                    "start_seed": args.start_seed,
                    "initial_state_hashes": initial_hashes,
                    "hold_lengths": hold_lengths(rate),
                    "telemetry": {
                        **telemetry,
                        "mean_pose_saturation_fraction": telemetry["pose_saturation_sum"] / max(1, telemetry["windows"]),
                    },
                }
                out.write_text(json.dumps(output, indent=2))
                print(key, f"{k}/{n}", output["cells"][key]["telemetry"], flush=True)
        env.close()
        del policy
        torch.cuda.empty_cache()

    expected = len(tasks) * len(rates) * len(arms)
    output["status"] = "COMPLETE" if len(output["cells"]) == expected else "INCOMPLETE"
    out.write_text(json.dumps(output, indent=2))
    print(f"[manifest] {output['status']} {len(output['cells'])}/{expected}", flush=True)


def self_check() -> None:
    assert hold_lengths(5) == [16.0, 16.0]
    assert hold_lengths(10) == [8.0, 8.0, 8.0, 8.0]
    assert hold_lengths(30) == [2.6, 2.6, 2.8] * 4
    assert hold_lengths(40) == [2.0] * 16
    torch.manual_seed(0)
    candidate = torch.randn(2, 4, 3)
    weights = torch.tensor([8.0, 8.0, 8.0, 8.0])
    lower = torch.full((3,), -0.5)
    upper = torch.full((3,), 0.5)
    target = torch.tensor([[1.0, -2.0, 0.0], [-3.0, 2.0, 1.0]])
    projected = weighted_box_sum_projection(candidate, weights, target, lower, upper)
    assert float((projected * weights.view(1, -1, 1)).sum(1).sub(target).abs().max()) < 2e-5
    assert float(projected.min()) >= -0.50001 and float(projected.max()) <= 0.50001
    print("self-check passed")


if __name__ == "__main__":
    if os.getenv("SELF_CHECK"):
        self_check()
    else:
        main()
