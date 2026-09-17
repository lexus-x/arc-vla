"""Final eval of the RL (GRPO Flow-SDE) pilot checkpoint on LIBERO-Spatial.

Same protocol as baseline_eval.py (reuses evaluate_multirate_honest.py's
MODELS/ARMS registration pattern, same TRIALS_PER_TASK=10 => n=100), but
points at rl_checkpoint_pilot instead of the SFT baseline checkpoint, and
runs a single seed (0) to match baseline seed0 for a same-seed comparison,
per the agreed 1-seed preliminary/existence-proof scope.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, CAMPAIGN)

TRIALS_PER_TASK = 10
SEED = 0
CKPT = "/home/user/Desktop/multi-rate/vla-rft/rl_checkpoint_pilot"
OUT_DIR = "baseline_results/rl_pilot_seed0"

import evaluate_multirate_honest as E  # noqa: E402

E.MODELS["rl_pilot"] = ("flow", CKPT, None, None)
E.ARMS["rl_pilot_native_20env"] = ("rl_pilot", 20, None, None)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

sys.argv = [
    "evaluate_multirate_honest.py",
    "--arms", "rl_pilot_native_20env",
    "--trials_per_task", str(TRIALS_PER_TASK),
    "--suite", "libero_spatial",
    "--out", OUT_DIR,
]
print(f"\n===== RL PILOT EVAL seed={SEED} ckpt={CKPT} =====", flush=True)
E.main()

result_path = Path(OUT_DIR) / "multirate_honest.json"
if not result_path.exists():
    result_path = Path("/home/user/Desktop/multi-rate/vla-vault") / OUT_DIR / "multirate_honest.json"
data = json.loads(result_path.read_text())
agg = data["rl_pilot_native_20env"]["aggregate"]
n = data["rl_pilot_native_20env"]["diagnostics"]["n"]

# Baseline comparators
baseline_summary = json.loads(Path("baseline_results/summary.json").read_text())
baseline_seed0_agg = baseline_summary["per_seed"]["0"]["aggregate"]
baseline_mean = baseline_summary["mean"]
baseline_std = baseline_summary["std"]

delta_vs_mean = agg - baseline_mean
delta_vs_seed0 = agg - baseline_seed0_agg
noise_floor = 2 * baseline_std

result = {
    "checkpoint": CKPT,
    "suite": "libero_spatial",
    "trials_per_task": TRIALS_PER_TASK,
    "seed": SEED,
    "rl_pilot_aggregate": agg,
    "rl_pilot_n": n,
    "baseline_seed0_aggregate": baseline_seed0_agg,
    "baseline_3seed_mean": baseline_mean,
    "baseline_3seed_std": baseline_std,
    "delta_vs_baseline_mean_pp": delta_vs_mean * 100,
    "delta_vs_baseline_seed0_pp": delta_vs_seed0 * 100,
    "noise_floor_2std_pp": noise_floor * 100,
    "clears_noise_floor_vs_mean": abs(delta_vs_mean) > noise_floor,
    "source": str(result_path),
}
Path("rl_eval_result.json").write_text(json.dumps(result, indent=2))
print(f"\n===== RL PILOT: {agg*100:.2f}% (n={n}) vs baseline mean {baseline_mean*100:.2f}%+/-{baseline_std*100:.2f}% (seed0: {baseline_seed0_agg*100:.2f}%) =====", flush=True)
print(f"Delta vs mean: {delta_vs_mean*100:+.2f}pp | Delta vs seed0: {delta_vs_seed0*100:+.2f}pp | Noise floor (2std): {noise_floor*100:.2f}pp | Clears floor: {abs(delta_vs_mean) > noise_floor}", flush=True)
print(json.dumps(result, indent=2))
