"""1-seed baseline eval of a suite's flow8300 SFT checkpoint, generalized from
baseline_eval.py to any of the 4 LIBERO suites via --suite / --ckpt.

Uses evaluate_multirate_honest.py's own --suite support (SUITE_WALLCLOCK dict
already defines the correct per-suite horizon: spatial 220, object 280,
goal 300, long 520 steps @20Hz -- using Spatial's budget on other suites
silently truncates them per that module's own documented warning).
"""
import argparse
import json
import sys
from pathlib import Path

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, CAMPAIGN)

DATASET_BY_SUITE = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_10": "lerobot/libero_10_image",
}

parser = argparse.ArgumentParser()
parser.add_argument("--suite", required=True, choices=list(DATASET_BY_SUITE))
parser.add_argument("--ckpt", required=True)
parser.add_argument("--trials_per_task", type=int, default=10)
parser.add_argument("--out", required=True)
args = parser.parse_args()

import evaluate_multirate_honest as E  # noqa: E402

arm_name = f"flow8300_{args.suite}_native_20env"
model_key = f"flow8300_{args.suite}"
E.MODELS[model_key] = ("flow", args.ckpt, None, None)
E.ARMS[arm_name] = (model_key, 20, None, None)
# CRITICAL: evaluate_multirate_honest.py resolves stats via
# MODEL_DATASET.get(model_key, DEFAULT_DATASET=libero_spatial) -- without this,
# a per-suite checkpoint silently gets Spatial's normalization stats, corrupting
# its action outputs (this exact bug produced 0.0%/2.0% "results" before being
# caught and fixed here; see PHASE4 report).
if args.suite != "libero_spatial":
    E.MODEL_DATASET[model_key] = DATASET_BY_SUITE[args.suite]

Path(args.out).mkdir(parents=True, exist_ok=True)
sys.argv = [
    "evaluate_multirate_honest.py",
    "--arms", arm_name,
    "--trials_per_task", str(args.trials_per_task),
    "--suite", args.suite,
    "--out", args.out,
]
print(f"===== BASELINE EVAL suite={args.suite} ckpt={args.ckpt} =====", flush=True)
E.main()

results_name = "multirate_honest.json" if args.suite == "libero_spatial" else f"multirate_{args.suite}.json"
result_path = Path(args.out) / results_name
data = json.loads(result_path.read_text())
agg = data[arm_name]["aggregate"]
n = data[arm_name]["diagnostics"]["n"]
print(f"RESULT suite={args.suite} aggregate={agg*100:.2f}% (n={n}) source={result_path}", flush=True)

summary = {"suite": args.suite, "ckpt": args.ckpt, "aggregate": agg, "n": n, "source": str(result_path)}
(Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))
