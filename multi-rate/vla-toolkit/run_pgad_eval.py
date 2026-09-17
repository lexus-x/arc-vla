"""Runs the PGAD toolkit's eval campaign: 2 new arms (gated, always_careful)
across the pre-existing PHASE6 8-task grid (spatial{7,8}, object/goal/10{0,1}),
n=20 episodes/task -- SAME seeds as the already-existing frozen baseline
(vla-rft/logs/campaign_*_frozen.json), via multi_task_campaign.seed_range, so
the new arms are seed-paired against a baseline that is NOT rerun here (BASELINE
stage reuses that existing, already-verified data; see FRAME.md's deadline-driven
reuse decision).

Worker-partitioned by SUITE (4 suites, 1 per worker at worker_count=4) so each
worker loads its suite's policy + progress head exactly once, matching this
project's existing run_multi_task_campaign.sh pattern (skip-if-done, continue
past a single failed episode logged not silently dropped, incremental save).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).parent
VLA_RFT = "/home/user/Desktop/multi-rate/vla-rft"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, VLA_RFT)

import suite_screen  # noqa: E402
from multi_task_campaign import CKPT_BY_SUITE, seed_range  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402

import pgad_core  # noqa: E402

SUITE_TASK_IDS = {
    "libero_spatial": [7, 8],
    "libero_object": [0, 1],
    "libero_goal": [0, 1],
    "libero_10": [0, 1],
}
SUITES_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
ARMS = ["gated", "always_careful"]
N_EPISODES_PER_ARM = 20  # matches the existing frozen baseline's n, for a fair paired comparison

OUT_DIR = ROOT / "pgad_results"
OUT_DIR.mkdir(exist_ok=True)
MANIFEST = ROOT / "pgad_campaign_manifest.jsonl"


def log_manifest(event: str, **kw):
    rec = {"event": event, "ts": datetime.now(timezone.utc).astimezone().isoformat(), **kw}
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(rec) + "\n")


def out_path(suite: str, task_id: int, arm: str) -> Path:
    return OUT_DIR / f"pgad_{suite}_task{task_id}_{arm}.json"


def run_combo(policy, preproc, head_bundle, resize_wh, screen, suite, task_id, arm):
    path = out_path(suite, task_id, arm)
    if path.exists():
        d = json.loads(path.read_text())
        if len(d.get("records", [])) >= N_EPISODES_PER_ARM:
            print(f"  skip {suite} task{task_id} {arm} (done)", flush=True)
            return

    env = screen._make_env(task_id)
    task_description = env.task_description
    head, feat_mean, feat_std, state_mean, state_std = head_bundle
    gate = pgad_core.ProgressGate(head, feat_mean, feat_std, state_mean, state_std, resize_wh, mode=arm)

    seeds = list(seed_range(suite, task_id, "frozen"))  # exact same seeds as the existing frozen baseline
    assert len(seeds) == N_EPISODES_PER_ARM

    records = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        try:
            out = pgad_core.rollout_episode_pgad(policy, preproc, env, task_description, seed, screen,
                                                  screen.MAX_STEPS, gate)
            rec = {"arm": arm, "episode": i, "seed": seed, **out}
        except Exception as e:
            rec = {"arm": arm, "episode": i, "seed": seed, "success": False, "error": str(e)}
            print(f"    [{suite} t{task_id} {arm} ep{i} seed={seed}] FAILED: {e}", flush=True)
        records.append(rec)
        print(f"    [{suite} t{task_id} {arm} ep{i}] seed={seed} success={rec.get('success')} "
              f"n_stalled={rec.get('n_stalled_decisions')}", flush=True)
        path.write_text(json.dumps({
            "meta": {"suite": suite, "task_id": task_id, "task_description": task_description,
                      "arm": arm, "ckpt": CKPT_BY_SUITE[suite], "n_episodes": N_EPISODES_PER_ARM,
                      "seed_range": [seeds[0], seeds[-1]], "elapsed_s": time.time() - t0},
            "records": records,
        }, indent=2))  # incremental save, survives a crash mid-task

    log_manifest("COMBO_DONE", suite=suite, task_id=task_id, arm=arm,
                  n_success=sum(1 for r in records if r.get("success")), n=len(records))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--worker_count", type=int, default=1)
    args = ap.parse_args()

    my_suites = [s for i, s in enumerate(SUITES_ORDER) if i % args.worker_count == args.worker_id]
    print(f"worker {args.worker_id}/{args.worker_count}: suites={my_suites}", flush=True)

    for suite in my_suites:
        try:
            screen = suite_screen.get_screen(suite)
            stats = LeRobotDatasetMetadata(screen.DATASET).stats
            policy, (preproc, postproc) = screen.load_policy("flow", CKPT_BY_SUITE[suite], stats)
            policy._rl_postprocessor = postproc
            policy.eval()
            resize_wh = policy.config.resize_imgs_with_padding
            head_bundle = pgad_core.load_head(ROOT / f"progress_head_{suite}.pt")

            for task_id in SUITE_TASK_IDS[suite]:
                for arm in ARMS:
                    print(f"=== {suite} task{task_id} {arm} ===", flush=True)
                    run_combo(policy, preproc, head_bundle, resize_wh, screen, suite, task_id, arm)

            del policy
            torch.cuda.empty_cache()
        except Exception:
            log_manifest("SUITE_FAILED", suite=suite, traceback=traceback.format_exc())
            print(f"SUITE {suite} FAILED:\n{traceback.format_exc()}", flush=True)

    print("worker done", flush=True)


if __name__ == "__main__":
    main()
