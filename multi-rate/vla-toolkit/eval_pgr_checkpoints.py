"""Post-training BINARY-success eval for the 4 candidate-G checkpoints
(vanilla/pgr x seed0/seed1), on the exact task grid they were trained on
(TASK_IDS=(3,5), libero_spatial's pinned pilot scope) -- deliberately reuses
multi_task_campaign.rollout_episode (same flow-SDE mechanics as training,
same family as the existing frozen-baseline eval used earlier this session
for PGAD), NOT the shaped training-time reward -- success is_success only, so
vanilla and pgr are compared on the metric that actually matters.

n=20 episodes/task x 2 tasks = 40 episodes/checkpoint. Fresh seed block
(9_000_000+), disjoint from every seed range used anywhere else in this
project (training used 1000+seed*1_000_000+ep_counter, capped well under
2,000,000 for these short runs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VLA_RFT = "/home/user/Desktop/multi-rate/vla-rft"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, VLA_RFT)

import suite_screen  # noqa: E402
from multi_task_campaign import rollout_episode  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
import torch  # noqa: E402

SUITE = "libero_spatial"
TASK_IDS = (3, 5)  # the exact scope train_grpo_pgr.py trained on
N_PER_TASK = 20
SEED_BASE = 9_000_000
CHECKPOINTS = {
    "vanilla_s0": ROOT / "rl_checkpoint/vanilla_s0",
    "vanilla_s1": ROOT / "rl_checkpoint/vanilla_s1",
    "pgr_s0": ROOT / "rl_checkpoint/pgr_s0",
    "pgr_s1": ROOT / "rl_checkpoint/pgr_s1",
}
OUT_DIR = ROOT / "pgr_eval_results"
OUT_DIR.mkdir(exist_ok=True)


def eval_checkpoint(name, ckpt_path):
    out_path = OUT_DIR / f"{name}.json"
    if out_path.exists() and len(json.loads(out_path.read_text())["records"]) >= N_PER_TASK * len(TASK_IDS):
        print(f"skip {name} (done)", flush=True)
        return

    screen = suite_screen.get_screen(SUITE)
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preproc, postproc) = screen.load_policy("flow", str(ckpt_path), stats)
    policy._rl_postprocessor = postproc
    policy.eval()

    records = []
    for task_id in TASK_IDS:
        env = screen._make_env(task_id)
        task_description = env.task_description
        for i in range(N_PER_TASK):
            seed = SEED_BASE + task_id * 1000 + i
            out = rollout_episode(policy, preproc, env, task_description, seed, screen, screen.MAX_STEPS)
            records.append({"task_id": task_id, "episode": i, "seed": seed, "success": bool(out["success"])})
            print(f"[{name} t{task_id} ep{i}] seed={seed} success={out['success']}", flush=True)
            out_path.write_text(json.dumps({"name": name, "ckpt": str(ckpt_path), "records": records}, indent=2))

    del policy
    torch.cuda.empty_cache()


def main():
    for name, ckpt in CHECKPOINTS.items():
        if not (ckpt / "config.json").exists():
            print(f"MISSING checkpoint: {ckpt}", flush=True)
            continue
        print(f"=== eval {name} ===", flush=True)
        eval_checkpoint(name, ckpt)
    print("done", flush=True)


if __name__ == "__main__":
    main()
