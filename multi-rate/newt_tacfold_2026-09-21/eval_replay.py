#!/usr/bin/env python3
"""Preregistered paired open-loop trajectory-reconstruction benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np

from eval_newt_tacfold import (
    ARMS,
    ACTION_DIMS,
    EXPECTED_SHA256,
    apply_arm,
    atomic_json,
    build_agent,
    checkpoint_path,
    load_suite,
    make_task_env,
    paired_seed,
)


HERE = Path(__file__).resolve().parent


def collect(agent, env, task: str, task_index: int, episode: int):
    import torch

    obs, _ = env.reset(seed=episode)
    agent._prev_mean.zero_()
    actions = []
    success = False
    step = 0
    done = False
    while not done:
        seed = paired_seed(task, episode, step)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        action = agent(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            t0=torch.tensor([step == 0]),
            eval_mode=True,
            task=torch.tensor([task_index], device="cuda:0"),
            mpc=True,
        )[0, :ACTION_DIMS].numpy()
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        success = bool(info["success"])
        done = bool(terminated or truncated)
        step += 1
    return np.asarray(actions, dtype=np.float32), success


def replay(env, actions: np.ndarray, episode: int) -> bool:
    env.reset(seed=episode)
    success = False
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        success = bool(info["success"])
        if terminated or truncated:
            break
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--task", default="all")
    parser.add_argument("--output", type=Path, default=HERE / "replay_results.json")
    args = parser.parse_args()

    suite = load_suite()
    tasks = suite if args.task == "all" else [args.task]
    if not set(tasks) <= set(suite):
        raise ValueError("task is not in the preregistered suite")

    checkpoint = checkpoint_path()
    agent, cfg = build_agent(checkpoint, horizon=3)
    results = json.loads(args.output.read_text()) if args.output.exists() else {
        "protocol": "PREREG_REPLAY.md",
        "checkpoint_sha256": EXPECTED_SHA256,
        "episodes_per_task": args.episodes,
        "tasks": {},
    }
    if results["episodes_per_task"] != args.episodes:
        raise ValueError("output file has a different episode count")

    for task in tasks:
        task_index = cfg.global_tasks.index(task)
        record = results["tasks"].setdefault(task, {
            "task_index": task_index,
            "success": {arm: [] for arm in ARMS},
        })
        env = make_task_env(cfg, task)
        try:
            completed = min(len(record["success"][arm]) for arm in ARMS)
            for episode in range(completed, args.episodes):
                native_actions, collected_success = collect(agent, env, task, task_index, episode)
                outcome = {}
                for arm in ARMS:
                    outcome[arm] = replay(env, apply_arm(native_actions, arm), episode)
                if outcome["native"] != collected_success:
                    raise RuntimeError(f"native replay was not deterministic: {task} seed {episode}")
                for arm in ARMS:
                    record["success"][arm].append(outcome[arm])
                atomic_json(args.output, results)
                print(f"[{task}] {episode + 1}/{args.episodes} " + " ".join(
                    f"{arm}={int(outcome[arm])}" for arm in ARMS
                ), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
