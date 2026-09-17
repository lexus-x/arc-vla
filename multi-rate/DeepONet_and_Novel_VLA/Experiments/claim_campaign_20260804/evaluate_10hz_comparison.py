"""10 Hz Slow Control Rate Comparison: Stock Flow vs DeepONet (til_s0) on LIBERO-Spatial."""

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
    parser.add_argument("--trials_per_task", type=int, default=10, help="Trials per task (default 10)")
    parser.add_argument("--out", default="eval_10hz_comparison_out", help="Output directory")
    args = parser.parse_args()

    models_config = {
        "flow_s0": ("flow", "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/paper_repro/Spatial/runs/flow_s0/checkpoints/30000"),
        "til_s0": ("deeponet", "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/runs/til_s0/checkpoints/8300")
    }

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    stats = LeRobotDatasetMetadata(DATASET).stats
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "spatial_10hz_comparison.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    rate_hz = 10
    task_ids = list(range(10))

    for name, (head, checkpoint) in models_config.items():
        if head == "deeponet":
            os.environ["DEEPONET_HEAD"] = "til"
            os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"
        else:
            os.environ.pop("DEEPONET_HEAD", None)
            os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "1"

        print(f"\n==========================================")
        print(f"LOADING ARM: {name} ({head}) @ 10 Hz")
        print(f"==========================================")

        policy, (preprocessor, postprocessor) = load_policy(head, checkpoint, stats)
        
        if head == "deeponet":
            from rate_integrated_deeponet import RateIntegratedDeepONetHead
            heads = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)]
            if heads:
                heads[0].set_rate(rate_hz)
                print(f"Configured DeepONet Rate Integration to {rate_hz} Hz")

        model_res = results.setdefault(name, {"per_task": {}, "aggregate": None})

        for t_id in task_ids:
            task_entry = model_res["per_task"].setdefault(str(t_id), {"episodes": [], "success_rate": 0.0})
            episodes = task_entry["episodes"]
            env = _make_env(t_id)
            task_entry["task"] = env.task_description

            for trial in range(len(episodes), args.trials_per_task):
                env.init_state_id = trial
                assert getattr(env, "init_state_id", None) == trial
                ep = _rollout(policy, preprocessor, postprocessor, env, env.task_description, 1000 + trial)
                episodes.append(ep)
                task_entry["success_rate"] = float(np.mean([item["success"] for item in episodes]))
                results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
                print(f"[{name} @ 10Hz] Task {t_id} Trial {trial:02d}: {'OK' if ep['success'] else 'x'} ({ep['steps']} steps)", flush=True)
            
            env.close()

        rates = [entry["success_rate"] for entry in model_res["per_task"].values()]
        model_res["aggregate"] = float(np.mean(rates))
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        
        del policy
        torch.cuda.empty_cache()

    print("\n10 HZ COMPARISON BENCHMARK COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
