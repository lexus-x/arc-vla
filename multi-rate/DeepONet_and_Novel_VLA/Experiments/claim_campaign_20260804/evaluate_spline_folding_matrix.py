"""Bug-Fixed 4-Way Matrix Comparison: Alias Folding vs Spline Interpolation for DeepONet & Flow Matching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import CubicSpline

from evaluate_height_screen import (
    load_policy, _make_env, _policy_input, _rollout, DATASET
)

def resample_spline(action_chunk: np.ndarray, target_rate: int, native_rate: int = 20) -> np.ndarray:
    """Resample action chunk using Cubic Spline Interpolation."""
    if action_chunk.ndim < 2 or action_chunk.shape[0] <= 1:
        return action_chunk
    orig_len = action_chunk.shape[0]
    target_len = max(2, int(orig_len * (target_rate / native_rate)))
    x_orig = np.linspace(0, 1, orig_len)
    x_target = np.linspace(0, 1, target_len)
    
    cs = CubicSpline(x_orig, action_chunk, axis=0)
    return cs(x_target)

def resample_linear(action_chunk: np.ndarray, target_rate: int, native_rate: int = 20) -> np.ndarray:
    """Resample action chunk using Linear Interpolation."""
    if action_chunk.ndim < 2 or action_chunk.shape[0] <= 1:
        return action_chunk
    orig_len = action_chunk.shape[0]
    target_len = max(2, int(orig_len * (target_rate / native_rate)))
    x_orig = np.linspace(0, 1, orig_len)
    x_target = np.linspace(0, 1, target_len)
    
    resampled = np.zeros((target_len, action_chunk.shape[1]), dtype=action_chunk.dtype)
    for c in range(action_chunk.shape[1]):
        resampled[:, c] = np.interp(x_target, x_orig, action_chunk[:, c])
    return resampled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials_per_task", type=int, default=2, help="Trials per task (default 2)")
    parser.add_argument("--rate", type=int, required=True, help="Target rate (20 or 40)")
    parser.add_argument("--out", default="matrix_eval_out", help="Output directory")
    args = parser.parse_args()

    models = {
        "til_s0": ("deeponet", "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/runs/til_s0/checkpoints/8300"),
        "flow_s0": ("flow", "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/paper_repro/Spatial/runs/flow_s0/checkpoints/30000")
    }

    rate = args.rate
    interventions = ["unmodified", "folding", "enhanced_folding", "spline", "linear"]

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    stats = LeRobotDatasetMetadata(DATASET).stats
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"matrix_comparison_{rate}hz.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for method in interventions:
        for model_name, (head, checkpoint) in models.items():
            if head == "flow" and method in ("folding", "enhanced_folding"):
                continue  # Alias folding is specific to DeepONet operator head

            arm_key = f"{model_name}_{method}_{rate}hz"
            if arm_key in results and results[arm_key].get("aggregate") is not None:
                print(f"Skipping {arm_key} (already completed: {results[arm_key]['aggregate']*100:.1f}%)")
                continue

            if head == "deeponet":
                os.environ["DEEPONET_HEAD"] = "til"
                os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"
            else:
                os.environ.pop("DEEPONET_HEAD", None)
                os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "1"

            print(f"\n==========================================")
            print(f"RUNNING PARALLEL WORKER: {arm_key} ({method.upper()}) @ {rate} Hz")
            print(f"==========================================")

            policy, (preprocessor, postprocessor) = load_policy(head, checkpoint, stats)
            _raw_select = policy.select_action
            
            if head == "deeponet" and method in ("folding", "enhanced_folding"):
                from rate_integrated_deeponet import RateIntegratedDeepONetHead
                heads = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)]
                if heads:
                    heads[0].set_rate(rate)
                    if method == "enhanced_folding":
                        if hasattr(heads[0], "set_enhanced_folding"):
                            heads[0].set_enhanced_folding(True)
                    print(f"Configured DeepONet {method.upper()} for {rate} Hz")

            if method == "spline" and rate != 20:
                def spline_action(batch):
                    action = _raw_select(batch)
                    action_np = action.cpu().numpy()
                    if action_np.ndim >= 2 and action_np.shape[0] > 1:
                        resampled = resample_spline(action_np, target_rate=rate, native_rate=20)
                        return torch.from_numpy(resampled).to(action.device)
                    return action
                policy.select_action = spline_action

            if method == "linear" and rate != 20:
                def linear_action(batch):
                    action = _raw_select(batch)
                    action_np = action.cpu().numpy()
                    if action_np.ndim >= 2 and action_np.shape[0] > 1:
                        resampled = resample_linear(action_np, target_rate=rate, native_rate=20)
                        return torch.from_numpy(resampled).to(action.device)
                    return action
                policy.select_action = linear_action

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

    print(f"\nPARALLEL WORKER FOR {rate} HZ COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
