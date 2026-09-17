"""Stage 5 (IMPLEMENT) of candidate D: short continued-SFT fine-tune of the
libero_10 checkpoint, toggling ONE thing between arms -- the per-frame
sampling WEIGHT -- so the comparison is a clean matched-compute ablation
(same steps, same batch size, same LR schedule, same optimizer, same data
pool; only which frames get oversampled differs).

Failure-mode taxonomy (Claude-labeled, see FAILURE_TAXONOMY.md): the
dominant, most systematic failure cluster is tasks {0, 1, 7} -- all three
share the identical "put both ITEM1 and ITEM2 in the basket" template with
small grocery items (soup can, sauce bottle, cheese box, butter), and account
for 18/42 (43%) of all collected failures with the most complete failure
signature (basket ends empty, not partial progress).

--arm targeted: frames from tasks {0,1,7} get weight HARD_WEIGHT (oversampled).
--arm uniform:  all frames weight 1.0 (plain uniform resampling, matched
                total step/batch count -- the causal control).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).parent
VLA_RFT = "/home/user/Desktop/multi-rate/vla-rft"
CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, VLA_RFT)
sys.path.insert(0, CAMPAIGN)

import suite_screen  # noqa: E402
from multi_task_campaign import CKPT_BY_SUITE  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from modeling_smolvla_ph import SmolVLAPHPolicy  # noqa: E402

DEV = "cuda"
SUITE = "libero_10"
HARD_TASK_IDS = (0, 1, 7)  # the "put both X and Y in basket" failure cluster
HARD_WEIGHT = 3.0
N_STEPS = 400
BATCH_SIZE = 32
HEAD_LR = 1e-5
BACKBONE_LR = 2e-6  # conservative continued-FT rate, ~10x below this project's
                     # own stage2-from-scratch backbone_lr=1e-5 (train.py), since
                     # we're nudging an already-converged checkpoint, not training fresh.
GRAD_CLIP = 10.0


def to_device(b):
    if torch.is_tensor(b):
        return b.to(DEV, non_blocking=True)
    if isinstance(b, dict):
        return {k: to_device(v) for k, v in b.items()}
    return b


def build_frame_weights(meta, screen, hard_task_ids, hard_weight):
    """One weight per frame index (matches LeRobotDataset's flat frame indexing)."""
    hard_descriptions = {screen._make_env(tid).task_description for tid in hard_task_ids}
    n_eps = len(meta.episodes)
    total = 0
    ranges = []
    for i in range(n_eps):
        info = meta.episodes[i]
        lo, hi = info["dataset_from_index"], info["dataset_to_index"]
        ranges.append((lo, hi, any(td in info["tasks"] for td in hard_descriptions)))
        total = max(total, hi)
    weights = np.ones(total, dtype=np.float64)
    n_hard_frames = 0
    for lo, hi, is_hard in ranges:
        if is_hard:
            weights[lo:hi] = hard_weight
            n_hard_frames += hi - lo
    print(f"[weights] {n_hard_frames}/{total} frames from hard tasks {hard_task_ids} "
          f"({100*n_hard_frames/total:.1f}%)", flush=True)
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["targeted", "uniform"], required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--log_json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    screen = suite_screen.get_screen(SUITE)
    ckpt = CKPT_BY_SUITE[SUITE]
    policy = SmolVLAPHPolicy.from_pretrained(ckpt, ph_enabled=False).to(DEV)

    meta = LeRobotDatasetMetadata(screen.DATASET)
    dt = resolve_delta_timestamps(policy.config, meta)
    dataset = LeRobotDataset(screen.DATASET, delta_timestamps=dt)
    preprocessor, _post = make_smolvla_pre_post_processors(policy.config, dataset_stats=meta.stats)

    if args.arm == "targeted":
        weights = build_frame_weights(meta, screen, HARD_TASK_IDS, HARD_WEIGHT)
    else:
        weights = np.ones(dataset.num_frames, dtype=np.float64)
    assert len(weights) == dataset.num_frames, f"{len(weights)} vs {dataset.num_frames}"
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def cyclic_loader():
        while True:
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                 num_workers=4, pin_memory=True, drop_last=True)
            for batch in loader:
                yield batch

    data_iter = cyclic_loader()

    policy.unfreeze_all()
    optimizer = torch.optim.AdamW(
        policy.param_groups(backbone_lr=BACKBONE_LR, head_lr=HEAD_LR),
        betas=(0.9, 0.95), weight_decay=1e-6,
    )

    policy.train()
    log = []
    t0 = time.time()
    for step in range(N_STEPS):
        batch = to_device(preprocessor(next(data_iter)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ld = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            (p for p in policy.parameters() if p.requires_grad), GRAD_CLIP)
        optimizer.step()

        if step % 20 == 0 or step == N_STEPS - 1:
            print(f"[{args.arm}] step {step:4d} loss={ld['flow_matching_loss']:.4f} "
                  f"grad_norm={float(gnorm):.3f} elapsed={time.time()-t0:.0f}s", flush=True)
        log.append({"step": step, "flow_matching_loss": ld["flow_matching_loss"],
                    "grad_norm": float(gnorm)})
        if step % 50 == 0:
            Path(args.log_json).write_text(json.dumps(log))

    Path(args.log_json).write_text(json.dumps(log, indent=2))
    Path(args.out_ckpt).mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(args.out_ckpt)
    print(f"Saved to {args.out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
