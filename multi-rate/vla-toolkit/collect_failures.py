"""Stage 1 of candidate D: collect FAILURE rollouts from the frozen libero_10
baseline across all 10 tasks, saving keyframes (start/25/50/75/near-end) +
task description + outcome for each failure, for later failure-mode labeling.

Fresh seed block, disjoint from every other seed range used in this project
(training used up to ~2,000,000 across 2 seeds; eval used 9,000,000+;
calibration/campaign probes used 50,000-90,000-ish ranges) -- using
20,000,000+ here.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent
VLA_RFT = "/home/user/Desktop/multi-rate/vla-rft"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, VLA_RFT)

import suite_screen  # noqa: E402
from multi_task_campaign import CKPT_BY_SUITE  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402
from flow_sde import sample_actions_flow_sde  # noqa: E402

SUITE = "libero_10"
TASK_IDS = list(range(10))
EPISODES_PER_TASK = 10
SEED_BASE = 20_000_000
NOISE_LEVEL = 0.1
KEYFRAME_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 0.95]

OUT_DIR = ROOT / "failure_keyframes"
OUT_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = ROOT / "failure_manifest.json"


@torch.no_grad()
def rollout_with_frames(policy, preproc, env, task_description, seed, screen, max_steps):
    policy.eval()
    policy._queues[ACTION] = deque([], maxlen=policy.config.n_action_steps)
    for key in list(policy._queues.keys()):
        if key != ACTION:
            policy._queues[key] = deque([], maxlen=policy.config.n_action_steps)

    observation, _ = env.reset(seed=seed)
    frames = []  # list of (img1 uint8 HWC numpy, img2 uint8 HWC numpy)
    success = False
    step = 0
    for step in range(1, max_steps + 1):
        if len(policy._queues[ACTION]) == 0:
            raw = screen._policy_input(observation, task_description)
            img1 = (raw["observation.images.image"].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img2 = (raw["observation.images.wrist_image"].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append((img1, img2))

            batch = preproc(raw)
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            proc_batch = policy._prepare_batch(dict(batch))
            policy._queues = populate_queues(policy._queues, proc_batch, exclude_keys=[ACTION])
            for k in proc_batch:
                if k in policy._queues and k != ACTION:
                    proc_batch[k] = torch.stack(list(policy._queues[k]), dim=1)

            images, img_masks = policy.prepare_images(proc_batch)
            state = policy.prepare_state(proc_batch)
            lang_tokens = proc_batch["observation.language.tokens"]
            lang_masks = proc_batch["observation.language.attention_mask"]

            actions_shape = (1, policy.model.config.chunk_size, policy.model.config.max_action_dim)
            noise = policy.model.sample_noise(actions_shape, "cuda")
            result = sample_actions_flow_sde(
                policy, images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, noise_level=NOISE_LEVEL, stochastic=True,
            )
            actions = result["actions"]
            original_action_dim = policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            policy._queues[ACTION].extend(actions.transpose(0, 1)[: policy.config.n_action_steps])

        action = policy._queues[ACTION].popleft()
        obs_action = action.unsqueeze(0) if action.dim() == 1 else action
        env_action = policy._rl_postprocessor(obs_action).to("cpu").float().numpy().reshape(-1)
        observation, _, terminated, truncated, info = env.step(env_action)
        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    return {"success": success, "steps": step, "frames": frames}


def main():
    screen = suite_screen.get_screen(SUITE)
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preproc, postproc) = screen.load_policy("flow", CKPT_BY_SUITE[SUITE], stats)
    policy._rl_postprocessor = postproc
    policy.eval()

    failures = []
    for task_id in TASK_IDS:
        env = screen._make_env(task_id)
        task_description = env.task_description
        for ep in range(EPISODES_PER_TASK):
            seed = SEED_BASE + task_id * 1000 + ep
            out = rollout_with_frames(policy, preproc, env, task_description, seed, screen, screen.MAX_STEPS)
            print(f"[task{task_id} ep{ep}] seed={seed} success={out['success']} steps={out['steps']} "
                  f"n_frames={len(out['frames'])}", flush=True)
            if not out["success"]:
                n = len(out["frames"])
                fail_id = f"t{task_id}_ep{ep}_seed{seed}"
                fdir = OUT_DIR / fail_id
                fdir.mkdir(exist_ok=True)
                saved = []
                for frac in KEYFRAME_FRACTIONS:
                    idx = min(int(frac * (n - 1)), n - 1) if n > 0 else 0
                    img1, img2 = out["frames"][idx]
                    p1 = fdir / f"frac{frac:.2f}_cam1.png"
                    p2 = fdir / f"frac{frac:.2f}_cam2.png"
                    Image.fromarray(img1).save(p1)
                    Image.fromarray(img2).save(p2)
                    saved.append({"frac": frac, "cam1": str(p1), "cam2": str(p2)})
                failures.append({
                    "fail_id": fail_id, "task_id": task_id, "task_description": task_description,
                    "seed": seed, "steps": out["steps"], "keyframes": saved,
                })
                MANIFEST_PATH.write_text(json.dumps(failures, indent=2))  # incremental save

    del policy
    torch.cuda.empty_cache()
    print(f"\nCollected {len(failures)} failures across {len(TASK_IDS)} tasks.", flush=True)
    print(f"Manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
