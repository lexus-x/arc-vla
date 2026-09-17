"""Fast PILOT (n=10/task, not the final claim) testing the opposite-sign
hypothesis from VERIFY.md: RAISE noise_level on a detected stall (explore out
of a bad mode) instead of lowering it. Only the 2 tasks with the largest
observed effect (either direction) in the first campaign: libero_10 task0
(gated dropped 7/20->3/20, the worst result) and spatial task8 (gated dropped
16/20->12/20). Uses the first 10 of each task's existing 20 frozen-baseline
seeds, so it's directly comparable to a matching 10-seed slice of the
already-existing frozen numbers -- no baseline rerun needed here either.

If this looks promising, the full n=20 x 8-task grid gets rerun with the new
setting for the real numbers; this script's result is explicitly a
go/no-go pilot, not evidence for the paper.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
VLA_RFT = "/home/user/Desktop/multi-rate/vla-rft"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, VLA_RFT)

import suite_screen  # noqa: E402
from multi_task_campaign import CKPT_BY_SUITE, seed_range  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402

import pgad_core  # noqa: E402

PILOT_TASKS = [("libero_10", 0), ("libero_spatial", 8)]
N_PILOT = 10
EXPLORE_NOISE = 0.3  # opposite sign from the failed NOISE_CAREFUL=0.02; ~3x the default 0.1


def main():
    results = {}
    for suite, task_id in PILOT_TASKS:
        screen = suite_screen.get_screen(suite)
        stats = LeRobotDatasetMetadata(screen.DATASET).stats
        policy, (preproc, postproc) = screen.load_policy("flow", CKPT_BY_SUITE[suite], stats)
        policy._rl_postprocessor = postproc
        policy.eval()
        resize_wh = policy.config.resize_imgs_with_padding
        head_bundle = pgad_core.load_head(ROOT / f"progress_head_{suite}.pt")

        env = screen._make_env(task_id)
        task_description = env.task_description
        seeds = list(seed_range(suite, task_id, "frozen"))[:N_PILOT]

        gate = pgad_core.ProgressGate(*head_bundle, resize_wh, mode="gated", stall_noise_level=EXPLORE_NOISE)
        successes = []
        t0 = time.time()
        for seed in seeds:
            out = pgad_core.rollout_episode_pgad(policy, preproc, env, task_description, seed, screen,
                                                  screen.MAX_STEPS, gate)
            successes.append(bool(out["success"]))
            print(f"[{suite} t{task_id} explore] seed={seed} success={out['success']} "
                  f"n_stalled={out['n_stalled_decisions']}/{out['n_decisions']}", flush=True)

        results[f"{suite}_t{task_id}"] = {
            "n": len(successes), "k": sum(successes), "elapsed_s": time.time() - t0,
        }
        del policy
        import torch
        torch.cuda.empty_cache()

    (ROOT / "pilot_explore_noise_result.json").write_text(json.dumps(results, indent=2))
    print("\n=== PILOT SUMMARY (n=10/task, explore_noise=0.3) ===")
    for k, v in results.items():
        print(f"{k}: {v['k']}/{v['n']} = {v['k']/v['n']:.2f}")


if __name__ == "__main__":
    main()
