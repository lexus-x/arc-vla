"""Suite-generalized RL checkpoint eval, mirroring baseline_eval_suite.py's
pattern (same evaluate_multirate_honest.py MODELS/ARMS registration, same
per-suite MODEL_DATASET fix) but pointed at an RL-trained checkpoint and
diffed against that suite's own baseline_eval_suite.py output.
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
parser.add_argument("--baseline_summary", required=True,
                     help="path to that suite's baseline_eval_suite.py summary.json")
parser.add_argument("--trials_per_task", type=int, default=10)
parser.add_argument("--out", required=True)
args = parser.parse_args()

import evaluate_multirate_honest as E  # noqa: E402

arm_name = f"rl_pilot_{args.suite}_native_20env"
model_key = f"rl_pilot_{args.suite}"
E.MODELS[model_key] = ("flow", args.ckpt, None, None)
E.ARMS[arm_name] = (model_key, 20, None, None)
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
print(f"===== RL EVAL suite={args.suite} ckpt={args.ckpt} =====", flush=True)
E.main()

results_name = "multirate_honest.json" if args.suite == "libero_spatial" else f"multirate_{args.suite}.json"
result_path = Path(args.out) / results_name
data = json.loads(result_path.read_text())
agg = data[arm_name]["aggregate"]
n = data[arm_name]["diagnostics"]["n"]

baseline = json.loads(Path(args.baseline_summary).read_text())
baseline_agg = baseline["aggregate"]
delta_pp = (agg - baseline_agg) * 100

result = {
    "suite": args.suite,
    "rl_ckpt": args.ckpt,
    "rl_aggregate": agg,
    "rl_n": n,
    "baseline_aggregate": baseline_agg,
    "baseline_source": args.baseline_summary,
    "delta_pp": delta_pp,
    "note": "1-seed preliminary, no significance claim possible (no baseline stddev for this suite)",
    "source": str(result_path),
}
(Path(args.out) / "rl_vs_baseline.json").write_text(json.dumps(result, indent=2))
print(f"RESULT suite={args.suite} RL={agg*100:.2f}% baseline={baseline_agg*100:.2f}% delta={delta_pp:+.2f}pp (n={n})", flush=True)
