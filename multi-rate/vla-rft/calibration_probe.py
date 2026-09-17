"""Calibration probe for candidate F ("predict per-task RL benefit before training").
Runs short FROZEN-policy rollout groups (same GROUP_SIZE as multi_task_campaign.py's
RL arms) on a task, computes baseline_success_rate + per-group reward std +
degenerate-group rate, and applies a pre-committed prediction rule -- BEFORE any RL
training happens on that task. This script's output IS the pre-registration content;
it must be run and its results locked to disk before the full RL campaign starts.

Reused unmodified: get_screen_cached, rollout_episode, CKPT_BY_SUITE (multi_task_campaign.py).

Prediction rule (grounded in this project's own prior findings, not arbitrary
thresholds):
  - PHASE5 (Collapse-Aware GRPO) Long-suite: 5/8=62.5% degenerate GRPO updates
    coincided with that suite's regression under the vanilla arm.
  - Qwen-VLA (arXiv:2605.30280): ~97.8% baseline success (near-ceiling) coincided
    with only +0.1pp RL gain (effectively flat).
  Rule:
    baseline_success_rate >= 0.90  -> REGRESS_OR_FLAT  (near-ceiling: little headroom,
                                       Qwen-VLA precedent)
    baseline_success_rate <= 0.10  -> REGRESS_OR_FLAT  (near-floor: same degenerate-
                                       group mechanism at the opposite end -- almost
                                       every group is all-failure, zero advantage)
    degenerate_group_rate >= 0.5   -> REGRESS_OR_FLAT  (PHASE5 Long-suite precedent,
                                       directly on the GRPO advantage-collapse signal)
    otherwise                      -> IMPROVE
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from multi_task_campaign import get_screen_cached, rollout_episode, CKPT_BY_SUITE  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402

GROUP_SIZE = 4       # matches multi_task_campaign.py's RL-arm group size
N_PROBE_GROUPS = 3   # cheap probe: 3 groups x 4 = 12 rollouts/task (vs 20 for a full arm)
PROBE_SEED_BASE = 900_000  # disjoint from multi_task_campaign's SEED_BASE=50000 block
                            # (that grid tops out well under 100000) and from train_grpo*
                            # pilot seed ranges (2000-9999 / 77000+/88000+) -- see that
                            # module's SEED_BASE comment.


def predict(baseline_success_rate: float, degenerate_group_rate: float) -> str:
    if baseline_success_rate >= 0.90 or baseline_success_rate <= 0.10 or degenerate_group_rate >= 0.5:
        return "REGRESS_OR_FLAT"
    return "IMPROVE"


def probe_task(suite: str, task_id: int) -> dict:
    screen = get_screen_cached(suite)
    max_steps = screen.MAX_STEPS
    ckpt = CKPT_BY_SUITE[suite]
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preproc, postproc) = screen.load_policy("flow", ckpt, stats)
    policy._rl_postprocessor = postproc
    policy.eval()

    env = screen._make_env(task_id)
    task_description = env.task_description

    group_stds = []
    all_success = []
    for group_idx in range(N_PROBE_GROUPS):
        rewards = []
        for g in range(GROUP_SIZE):
            seed = PROBE_SEED_BASE + task_id * 1000 + group_idx * GROUP_SIZE + g
            out = rollout_episode(policy, preproc, env, task_description, seed, screen, max_steps)
            r = 1.0 if out["success"] else 0.0
            rewards.append(r)
            all_success.append(r)
            print(f"    [{suite} t{task_id} g{group_idx} r{g}] seed={seed} success={out['success']}", flush=True)
        group_stds.append(float(np.std(rewards)))

    del policy
    torch.cuda.empty_cache()

    baseline_success_rate = float(np.mean(all_success))
    degenerate_group_rate = float(np.mean([s < 1e-8 for s in group_stds]))
    pred = predict(baseline_success_rate, degenerate_group_rate)
    return {
        "suite": suite,
        "task_id": task_id,
        "n_rollouts": len(all_success),
        "n_groups": N_PROBE_GROUPS,
        "baseline_success_rate": baseline_success_rate,
        "group_stds": group_stds,
        "degenerate_group_rate": degenerate_group_rate,
        "prediction": pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_json", required=True, help="JSON list of [suite, task_id] pairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    grid = json.loads(Path(args.grid_json).read_text())

    results = []
    for suite, task_id in grid:
        print(f"=== probing {suite} task_id={task_id} ===", flush=True)
        r = probe_task(suite, task_id)
        print(f"    baseline_success_rate={r['baseline_success_rate']:.3f} "
              f"degenerate_group_rate={r['degenerate_group_rate']:.3f} "
              f"prediction={r['prediction']}", flush=True)
        results.append(r)
        Path(args.out).write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "done": False,
            "results": results,
        }, indent=2))  # incremental save so partial progress survives a crash

    Path(args.out).write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "done": True,
        "results": results,
    }, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
