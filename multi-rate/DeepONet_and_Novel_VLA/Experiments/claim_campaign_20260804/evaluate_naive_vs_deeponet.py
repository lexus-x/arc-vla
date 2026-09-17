"""Head-to-Head Benchmark: Well-Known Naive Rate Rescaling (20/f) vs Trajectory-Integrated DeepONet (til_s0)."""

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
    parser.add_argument("--trials_per_task", type=int, default=2, help="Trials per task (default 2)")
    parser.add_argument("--rate", type=int, default=40, help="Target rate (40 Hz)")
    parser.add_argument("--out", default="naive_vs_deeponet_out", help="Output directory")
    args = parser.parse_args()

    models = {
        "naive_legacy_flow_40hz": ("flow", "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/paper_repro/Spatial/runs/flow_s0/checkpoints/30000"),
        "til_s0_deeponet_40hz": ("deeponet", "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/runs/til_s0/checkpoints/8300")
    }

    rate = args.rate
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    stats = LeRobotDatasetMetadata(DATASET).stats
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"naive_vs_deeponet_{rate}hz.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for arm_key, (head, checkpoint) in models.items():
        if arm_key in results and results[arm_key].get("aggregate") is not None:
            print(f"Skipping {arm_key} (already completed)")
            continue

        if head == "deeponet":
            os.environ["DEEPONET_HEAD"] = "til"
            os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"
        else:
            os.environ.pop("DEEPONET_HEAD", None)
            os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "1"

        print(f"\n==========================================")
        print(f"RUNNING ARM: {arm_key} @ {rate} Hz")
        print(f"==========================================")

        policy, (preprocessor, postprocessor) = load_policy(head, checkpoint, stats)
        _raw_select = policy.select_action
        
        if head == "deeponet":
            from rate_integrated_deeponet import RateIntegratedDeepONetHead
            heads = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)]
            if heads:
                heads[0].set_rate(rate)
                print(f"Configured DeepONet Rate Integration for {rate} Hz")
        else:
            # Naive legacy scaling wrapper (scale = 20 / rate)
            def naive_scale_action(batch):
                action = _raw_select(batch)
                scale = 20.0 / rate
                return action * scale
            policy.select_action = naive_scale_action
            print(f"Configured Naive Legacy Rescaling (scale = {20.0/rate:.2f})")

        model_res = results.setdefault(arm_key, {"per_task": {}, "aggregate": None})

        for t_id in range(10):
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
                print(f"[{arm_key}] Task {t_id} Trial {trial:02d}: {'OK' if ep['success'] else 'x'} ({ep['steps']} steps)", flush=True)
            
            env.close()

        rates_list = [entry["success_rate"] for entry in model_res["per_task"].values()]
        model_res["aggregate"] = float(np.mean(rates_list))
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"--> ARM {arm_key} AGGREGATE RESULT: {model_res['aggregate']*100:.1f}%\n")
        
        del policy
        torch.cuda.empty_cache()

    print(f"\nHEAD-TO-HEAD BENCHMARK FOR {rate} HZ COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
