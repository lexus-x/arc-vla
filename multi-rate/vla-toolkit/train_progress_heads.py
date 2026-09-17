"""Trains one progress-regression head per LIBERO suite, generalizing
vla-rft/progress_estimator.py (which trained a single head on ONE task's
demos only) to the PGAD toolkit's 8-task grid (2 tasks/suite, matching
PHASE6_PREREGISTRATION.md's spatial{7,8}/object{0,1}/goal{0,1}/libero_10{0,1}).

Why per-suite (not one pooled multi-suite head): each suite uses a separately
fine-tuned checkpoint (CKPT_BY_SUITE), so frozen backbone features are not
guaranteed comparable across suites -- pooling would mix incomparable feature
spaces. Per-suite, pooled across that suite's 2 target tasks, is the cheapest
scoping that is still genuinely multi-task (not single-task like the original)
and keeps backbone/feature-space consistent. Documented scope decision, not a
correctness shortcut -- see FRAME.md / 2-day deadline.

Reuses (imported, unmodified): ProgressHead, embed_batch, train_head from
vla-rft/progress_estimator.py. Only collect_demo_dataset is regeneralized here
(the original hardcodes a single TASK_DESCRIPTION/SUITE/CKPT at module level).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

VLA_RFT = "/home/user/Desktop/multi-rate/vla-rft"
sys.path.insert(0, VLA_RFT)

import progress_estimator as pe  # noqa: E402 -- reused module (ProgressHead, embed_batch)
import pgad_core  # noqa: E402 -- corrected trainer (normalizes feats too, see pgad_core docstring)
import suite_screen  # noqa: E402
from multi_task_campaign import CKPT_BY_SUITE  # noqa: E402 -- reused, per-suite checkpoints
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

HERE = Path(__file__).parent
SUITE_TASK_IDS = {
    "libero_spatial": [7, 8],
    "libero_object": [0, 1],
    "libero_goal": [0, 1],
    "libero_10": [0, 1],
}
VAL_EPISODE_FRACTION = 0.2
SPLIT_SEED = 0


def collect_demo_dataset_multi(screen, policy, task_descriptions):
    """Generalizes progress_estimator.collect_demo_dataset to pool demos from
    multiple task descriptions within one suite's dataset."""
    ds = LeRobotDataset(screen.DATASET)
    meta = ds.meta
    ep_ids = [
        i for i in range(len(meta.episodes))
        if any(td in meta.episodes[i]["tasks"] for td in task_descriptions)
    ]
    rng = np.random.RandomState(SPLIT_SEED)
    ep_ids_shuffled = ep_ids.copy()
    rng.shuffle(ep_ids_shuffled)
    n_val = max(1, int(round(len(ep_ids_shuffled) * VAL_EPISODE_FRACTION)))
    val_eps = set(ep_ids_shuffled[:n_val])
    train_eps = set(ep_ids_shuffled[n_val:])

    def gather(ep_set):
        imgs1, imgs2, states, progress, ep_of_frame = [], [], [], [], []
        for ep in ep_set:
            info = meta.episodes[ep]
            lo, hi = info["dataset_from_index"], info["dataset_to_index"]
            length = hi - lo
            for j, idx in enumerate(range(lo, hi)):
                frame = ds[idx]
                imgs1.append(frame["observation.images.image"])
                imgs2.append(frame["observation.images.wrist_image"])
                states.append(frame["observation.state"].numpy())
                progress.append(j / (length - 1) if length > 1 else 1.0)
                ep_of_frame.append(ep)
        return imgs1, imgs2, np.stack(states).astype(np.float32), np.array(progress, dtype=np.float32), np.array(ep_of_frame)

    resize_wh = policy.config.resize_imgs_with_padding
    result = {}
    for split, ep_set in (("train", train_eps), ("val", val_eps)):
        imgs1, imgs2, states, progress, ep_of_frame = gather(ep_set)
        feats = pe.embed_batch(policy, imgs1, imgs2, resize_wh)
        result[split] = {"feats": feats, "states": states, "progress": progress, "ep": ep_of_frame}
        print(f"  [{split}] {len(ep_set)} episodes, {len(progress)} frames", flush=True)
    result["n_train_ep"] = len(train_eps)
    result["n_val_ep"] = len(val_eps)
    return result


def load_backbone_for(suite: str):
    screen = suite_screen.get_screen(suite)
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preproc, postproc) = screen.load_policy("flow", CKPT_BY_SUITE[suite], stats)
    policy._rl_postprocessor = postproc
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return screen, policy, preproc


def main():
    all_stats = {}
    for suite, task_ids in SUITE_TASK_IDS.items():
        print(f"=== {suite} (tasks {task_ids}) ===", flush=True)
        screen, policy, preproc = load_backbone_for(suite)
        task_descriptions = [screen._make_env(tid).task_description for tid in task_ids]
        print(f"  task descriptions: {task_descriptions}", flush=True)

        data = collect_demo_dataset_multi(screen, policy, task_descriptions)
        head, feat_mean, feat_std, state_mean, state_std, train_stats = pgad_core.train_head_normalized(data)
        pgad_core.save_head(HERE / f"progress_head_{suite}.pt", head, feat_mean, feat_std, state_mean, state_std)
        print(f"  held-out r={train_stats['held_out_overall_pearson_r']:.4f} "
              f"per-ep r={train_stats['held_out_per_episode_pearson_r_mean']}", flush=True)
        all_stats[suite] = {"task_ids": task_ids, "task_descriptions": task_descriptions, **train_stats}

        del policy
        import torch
        torch.cuda.empty_cache()

    (HERE / "progress_heads_train_stats.json").write_text(json.dumps(all_stats, indent=2))
    print("\nWrote progress_heads_train_stats.json", flush=True)


if __name__ == "__main__":
    main()
