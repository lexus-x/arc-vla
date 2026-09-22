#!/usr/bin/env python3
"""Paired closed-loop Newt evaluation for the preregistered Panda20 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
NEWT_ROOT = HERE.parent / "newt-src" / "tdmpc2"
RESAMPLE_ROOT = HERE.parent / "full_grid_2026-09-07"
sys.path[:0] = [str(NEWT_ROOT), str(RESAMPLE_ROOT)]

from resample_math import decimate_and_resample  # noqa: E402


ARMS = ("native", "zoh", "spline", "bspline", "tac_fold")
K = 2
PLAN_HORIZON = 8
CONTINUOUS_DIMS = 6
ACTION_DIMS = 7
EXPECTED_SHA256 = "54d76a7dc98d41fc882d446a72892e28e779f7a2f2f17c096ff336f1fe4a2232"


def load_suite() -> list[str]:
    with (HERE / "suite.json").open() as stream:
        tasks = json.load(stream)["tasks"]
    assert len(tasks) == 20 and len(set(tasks)) == 20
    return tasks


def checkpoint_path() -> Path:
    matches = list((HERE / "checkpoints").glob("models--nicklashansen--newt/snapshots/*/soup-20M-default.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one soup-20M-default.pt, found {len(matches)}")
    path = matches[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("checkpoint SHA-256 mismatch")
    return path


def paired_seed(task: str, episode: int, replan: int) -> int:
    digest = hashlib.sha256(f"{task}:{episode}:{replan}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def apply_arm(plan: np.ndarray, arm: str) -> np.ndarray:
    """Transform one clipped Tx7 delta-pose trajectory; causal-hold the gripper."""
    plan = np.asarray(plan, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[1] != ACTION_DIMS:
        raise ValueError(f"expected (T, {ACTION_DIMS}), got {plan.shape}")
    plan = np.clip(plan, -1, 1)
    if arm == "native":
        return plan.copy()
    continuous = decimate_and_resample(plan[:, :CONTINUOUS_DIMS], K, arm)
    gripper = np.repeat(plan[::K, CONTINUOUS_DIMS:], K, axis=0)[:len(plan)]
    return np.concatenate((continuous, gripper), axis=1).astype(np.float32)


def build_agent(checkpoint: Path, horizon: int = PLAN_HORIZON):
    import torch
    from omegaconf import OmegaConf
    from common.world_model import WorldModel
    from config import Config, parse_cfg
    from tdmpc2 import TDMPC2

    raw = OmegaConf.structured(Config(
        task="soup",
        checkpoint=str(checkpoint),
        model_size="L",
        obs="state",
        compile=False,
        horizon=horizon,
        num_envs=1,
        rank=0,
    ))
    cfg = parse_cfg(raw)
    cfg.num_envs = 1
    cfg.obs_shape = {"state": (128,)}
    cfg.action_dim = 16
    cfg.episode_length = 50
    cfg.world_size = 1
    model = WorldModel(cfg).to("cuda:0")
    agent = TDMPC2(model, cfg)
    agent.load(str(checkpoint))
    agent.eval()
    return agent, cfg


def make_task_env(global_cfg, task: str):
    from envs.maniskill import make_env
    from envs.wrappers.vectorized_multitask import VecWrapper

    cfg = deepcopy(global_cfg)
    cfg.task = task
    cfg.tasks = [task]
    cfg.num_tasks = cfg.num_envs = 1
    cfg.child_env = True
    return VecWrapper(make_env(cfg))


def run_episode(agent, env, task: str, task_index: int, episode: int, arm: str, official_native: bool = False):
    import torch

    obs, _ = env.reset(seed=episode)
    agent._prev_mean.zero_()
    final_success = False
    replan = 0
    saturation = []
    done = False
    while not done:
        seed = paired_seed(task, episode, replan)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        task_tensor = torch.tensor([task_index], device="cuda:0")
        action = agent(obs_tensor, t0=torch.tensor([replan == 0]), eval_mode=True, task=task_tensor, mpc=True)
        if official_native:
            obs, _, terminated, truncated, info = env.step(action[0].numpy())
            final_success = bool(info["success"])
            done = bool(terminated or truncated)
            replan += 1
            continue
        plan = agent._prev_mean[0, :PLAN_HORIZON, :ACTION_DIMS].detach().cpu().numpy()
        saturation.append(float((np.abs(plan[:, :CONTINUOUS_DIMS]) >= 1).mean()))
        for action in apply_arm(plan, arm):
            obs, _, terminated, truncated, info = env.step(action)
            final_success = bool(info["success"])
            done = bool(terminated or truncated)
            if done:
                break
        replan += 1
    return final_success, float(np.mean(saturation)) if saturation else 0.0


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", help="suite task or all")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--official-native", action="store_true", help="calibrate the checkpoint with Newt's official horizon-3, replan-every-step evaluator")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suite = load_suite()
    tasks = suite if args.task == "all" else [args.task]
    if not set(tasks) <= set(suite):
        raise ValueError("task is not in the preregistered suite")
    arms = tuple(args.arms.split(","))
    if not arms or not set(arms) <= set(ARMS):
        raise ValueError(f"arms must be drawn from {ARMS}")
    if args.official_native and arms != ("native",):
        raise ValueError("--official-native only supports --arms native")
    ckpt = checkpoint_path()
    print(json.dumps({"tasks": tasks, "arms": arms, "episodes": args.episodes, "checkpoint": str(ckpt)}, indent=2))
    if args.dry_run:
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the official Newt inference implementation")
    agent, cfg = build_agent(ckpt, horizon=3 if args.official_native else PLAN_HORIZON)
    results = json.loads(args.output.read_text()) if args.output.exists() else {
        "protocol": "PREREG.md",
        "checkpoint_sha256": EXPECTED_SHA256,
        "episodes_per_task": args.episodes,
        "official_native": args.official_native,
        "tasks": {},
    }
    if results.get("official_native", False) != args.official_native:
        raise ValueError("output file belongs to a different native protocol")
    for task in tasks:
        task_index = cfg.global_tasks.index(task)
        record = results["tasks"].setdefault(task, {
            "task_index": task_index,
            "success": {arm: [] for arm in arms},
            "plan_saturation": {arm: [] for arm in arms},
        })
        env = make_task_env(cfg, task)
        try:
            completed = min(len(record["success"].setdefault(arm, [])) for arm in arms)
            for episode in range(completed, args.episodes):
                episode_out = {}
                for arm in arms:
                    success, saturation = run_episode(agent, env, task, task_index, episode, arm, args.official_native)
                    record["success"][arm].append(success)
                    record["plan_saturation"].setdefault(arm, []).append(saturation)
                    episode_out[arm] = int(success)
                atomic_json(args.output, results)
                print(f"[{task}] {episode + 1}/{args.episodes} " + " ".join(f"{a}={episode_out[a]}" for a in arms), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
