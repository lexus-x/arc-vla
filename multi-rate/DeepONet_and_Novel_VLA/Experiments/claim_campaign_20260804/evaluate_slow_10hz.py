"""Pinned 10 Hz Slow-Rate Retargeting Evaluation Runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

from evaluate_height_screen import (
    load_policy, _make_env, _policy_input, _rollout, DATASET
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rate_hz", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--task_id", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ["DEEPONET_HEAD"] = "til"
    os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"
    
    parts = args.model.split("=")
    name, head, checkpoint = parts[0], parts[1], parts[2]
    
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    stats = LeRobotDatasetMetadata(DATASET).stats
    
    print(f"Loading model {name} ({head}) from {checkpoint}...")
    policy, (preprocessor, postprocessor) = load_policy(head, checkpoint, stats)
    
    # Configure rate integration to target rate
    from rate_integrated_deeponet import RateIntegratedDeepONetHead
    heads = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)]
    if heads:
        heads[0].set_rate(args.rate_hz)
        print(f"Set DeepONet Head Rate Integration to {args.rate_hz} Hz")

    env = _make_env(args.task_id)
    task_desc = env.task_description
    print(f"Evaluating {name} at {args.rate_hz} Hz on Task {args.task_id} ('{task_desc}') for {args.n_trials} trials...")
    
    results = []
    for trial in range(args.n_trials):
        env.init_state_id = trial
        ep = _rollout(policy, preprocessor, postprocessor, env, task_desc, 1000 + trial)
        results.append(ep)
        print(f"[{name} @ {args.rate_hz}Hz] Task {args.task_id} Trial {trial:02d}: {'OK' if ep['success'] else 'x'} ({ep['steps']} steps)")

    succ_count = sum(1 for ep in results if ep["success"])
    succ_rate = succ_count / len(results)
    print(f"\n==========================================")
    print(f"FINAL RESULT: {name} @ {args.rate_hz} Hz (Task {args.task_id}): {succ_count}/{len(results)} = {succ_rate*100:.1f}%")
    print(f"==========================================")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_data = {
        "model": name,
        "rate_hz": args.rate_hz,
        "task_id": args.task_id,
        "trials": results,
        "success_rate": succ_rate,
        "n_trials": len(results)
    }
    (out_dir / "slow_rate_results.json").write_text(json.dumps(res_data, indent=2))

if __name__ == "__main__":
    main()
