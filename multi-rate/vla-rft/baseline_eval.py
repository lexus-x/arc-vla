"""3-seed baseline reproduction of flow8300_s0 on LIBERO-Spatial.

Reuses the vault's own evaluate_multirate_honest.py machinery exactly as
run_matched8300_spatial.sh does (same MODELS/ARMS registration pattern), but
loops it under 3 different global torch/numpy/random seeds to measure OUR OWN
eval-time noise floor for this exact checkpoint, per this project's standing
rule that a claimed RL delta must exceed ~2x this stddev.

Note on what "seed" varies here: evaluate_multirate_honest.py pins
env.init_state_id = trial and env.reset(seed=1000+trial) deterministically per
trial (see eval-harness-traps.md's "init_state_id drift bias" fix) -- so
init states are IDENTICAL across our 3 runs by design (matching the vault's
own protocol). What differs across our 3 seeds is the policy's own
flow-matching sampling noise (torch.randn calls inside sample_actions are not
seeded by the harness), which is exactly the source of stochasticity our
Flow-SDE RL eval will also exhibit -- so this measures the right thing.

Reduced scale vs the vault's canonical --trials_per_task 30 (n=300): using 10
here (n=100 per seed on libero_spatial's 10 tasks) to fit this pilot's time
budget. Documented explicitly, not hidden.
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
SEEDS = [0, 1, 2]
OUT_ROOT = Path("/home/user/Desktop/multi-rate/vla-rft/baseline_results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

import evaluate_multirate_honest as E  # noqa: E402

E.MODELS["flow8300"] = (
    "flow",
    "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300",
    None,
    None,
)
E.ARMS["flow8300_native_20env"] = ("flow8300", 20, None, None)

per_seed_results = {}
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    out_dir = f"baseline_results/seed{seed}"
    sys.argv = [
        "evaluate_multirate_honest.py",
        "--arms", "flow8300_native_20env",
        "--trials_per_task", str(TRIALS_PER_TASK),
        "--suite", "libero_spatial",
        "--out", out_dir,
    ]
    print(f"\n===== BASELINE EVAL seed={seed} =====", flush=True)
    E.main()

    result_path = Path("/home/user/Desktop/multi-rate/vla-vault") / out_dir / "multirate_honest.json"
    if not result_path.exists():
        # E.main() writes relative to CWD (the vla-rft dir if we cd'd there); check both.
        alt = Path(out_dir) / "multirate_honest.json"
        result_path = alt if alt.exists() else result_path
    data = json.loads(result_path.read_text())
    agg = data["flow8300_native_20env"]["aggregate"]
    n = data["flow8300_native_20env"]["diagnostics"]["n"]
    per_seed_results[seed] = {"aggregate": agg, "n": n, "source": str(result_path)}
    print(f"seed={seed}: aggregate={agg*100:.2f}% (n={n})", flush=True)

aggs = [v["aggregate"] for v in per_seed_results.values()]
mean = float(np.mean(aggs))
std = float(np.std(aggs, ddof=1)) if len(aggs) > 1 else 0.0

summary = {
    "checkpoint": "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300",
    "suite": "libero_spatial",
    "trials_per_task": TRIALS_PER_TASK,
    "seeds": SEEDS,
    "per_seed": per_seed_results,
    "mean": mean,
    "std": std,
    "std_ddof1_n_seeds": len(aggs),
}
(OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n===== BASELINE SUMMARY: {mean*100:.2f}% +/- {std*100:.2f}% (n={len(aggs)} seeds) =====", flush=True)
print(json.dumps(summary, indent=2))
