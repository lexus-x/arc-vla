"""Self-supervised progress/reward estimator for TT-VLA-style test-time RL.

TT-VLA (arXiv:2601.06748) replaces ground-truth is_success reward with a learned
dense progress estimator, since real deployment has no ground truth. Everything
else in this vault so far has cheated and used is_success directly. This file
closes that gap: a frozen-backbone + small-MLP-head regressor trained on the
standard self-supervised time-index proxy (progress(t) = t/(T-1) on successful
demos), validated two ways (held-out demo correlation, and success/fail trend
separation on fresh rollouts under the exact smoke-test protocol).

Backbone reuse (no new vision encoder): policy.model.vlm_with_expert.embed_image
-- the same frozen SigLIP+connector path SmolVLA's own prefix embedding uses
(see modeling_smolvla.py:654's embed_prefix call) -- mean-pooled over the 64
patch tokens per camera (2 cams -> 1920-dim) concatenated with the raw 8-dim
proprio state (1928-dim total). Only a 1928->512->128->1 MLP head is trained;
the backbone is called under torch.no_grad() and never touched.

Training data: lerobot/libero_spatial_image (via suite_screen.get_screen's
DATASET), filtered to the episodes whose task text matches TASK_ID=0's
benchmark description ("...between the plate and the ramekin...") -- NOTE the
HF dataset's own task_index numbering does NOT match the LIBERO benchmark's
task_id numbering (verified: benchmark task_id=0 is HF dataset task_index=4),
so filtering is done by matching task description text, not by task_index.
All 45 episodes for this task are expert demos (LIBERO's dataset has no
recorded failures), so no success-filtering is needed -- every episode gets a
clean 0->1 progress label by construction.

Rollout validation reuses smoke_naive_ttt_safety.py's exact rollout mechanics
(same checkpoint, same task_id=0 env, same flow-SDE stochastic sampler,
same NOISE_LEVEL=0.1) via its own copy of the rollout loop (this file does not
import from or edit smoke_naive_ttt_safety.py, per the no-touch instruction).
smoke_results_v3.json only stored per-episode SCALAR outcomes (success,
rho_safe, ...) -- it never persisted per-frame images/transitions -- so there
is no cached frame data to "load"; the only honest way to check the estimator
against real rollout trajectories is to re-collect a handful of fresh episodes
under the identical protocol (same checkpoint/task/config) and report what
actually happens, which is what this file does. Seeds are chosen from
smoke_results_v3.json's arm_A_frozen records that were labeled success/fail
there, but the flow-SDE sampler draws fresh stochastic noise each run (only
the env reset is seeded), so a fresh run's outcome for a given seed can differ
from the original recorded label -- this file reports the FRESH outcome it
actually observes, not the old label.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, CAMPAIGN)

import suite_screen  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402

from flow_sde import sample_actions_flow_sde  # noqa: E402  -- reused unmodified

DEVICE = "cuda"
CKPT = "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300"
SUITE = "libero_spatial"
TASK_ID = 0
TASK_DESCRIPTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"
NOISE_LEVEL = 0.1  # matches smoke_naive_ttt_safety.py

HERE = Path(__file__).parent
HEAD_PATH = HERE / "progress_head.pt"
EVAL_PATH = HERE / "progress_estimator_eval.json"

VAL_EPISODE_FRACTION = 0.2
SPLIT_SEED = 0
FEAT_DIM_PER_CAM = 960
STATE_DIM = 8
IN_DIM = 2 * FEAT_DIM_PER_CAM + STATE_DIM

# rollout validation: seeds pulled from smoke_results_v3.json's arm_A_frozen,
# a mix of originally-labeled-success + originally-labeled-fail seeds (see module
# docstring for why these are RE-collected, not loaded from that file). Extended
# from an initial n=6 to n=10 after the first pass surfaced a false positive
# (seed 2002) worth checking wasn't a one-off -- see progress_estimator_eval.json.
VAL_ROLLOUT_SEEDS = [2000, 2001, 2003, 2002, 2005, 2007, 2004, 2006, 2010, 2012]
VAL_ROLLOUT_ORIG_LABEL = {
    2000: True, 2001: True, 2003: True, 2002: False, 2005: False, 2007: False,
    2004: True, 2006: True, 2010: False, 2012: False,
}


class ProgressHead(nn.Module):
    def __init__(self, in_dim=IN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def load_backbone():
    screen = suite_screen.get_screen(SUITE)
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preproc, postproc) = screen.load_policy("flow", CKPT, stats)
    policy._rl_postprocessor = postproc
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return screen, policy, preproc


@torch.no_grad()
def embed_batch(policy, imgs1, imgs2, resize_wh, batch_size=32):
    """imgs1/imgs2: list of (3,H,W) float32 [0,1] CPU tensors (one cam each).
    Returns (N, 2*FEAT_DIM_PER_CAM) float32 numpy array, pooled over patches."""
    out = []
    for i in range(0, len(imgs1), batch_size):
        b1 = torch.stack(imgs1[i:i + batch_size]).to(DEVICE)
        b2 = torch.stack(imgs2[i:i + batch_size]).to(DEVICE)
        b1 = resize_with_pad(b1, *resize_wh, pad_value=0) * 2.0 - 1.0
        b2 = resize_with_pad(b2, *resize_wh, pad_value=0) * 2.0 - 1.0
        f1 = policy.model.vlm_with_expert.embed_image(b1).float().mean(dim=1)  # (b, 960)
        f2 = policy.model.vlm_with_expert.embed_image(b2).float().mean(dim=1)
        out.append(torch.cat([f1, f2], dim=1).cpu())
    return torch.cat(out, dim=0).numpy()


def collect_demo_dataset(screen, policy):
    """Load libero_spatial demos for TASK_ID=0, split by episode into train/val,
    extract frozen features + raw state + progress label for every frame."""
    ds = LeRobotDataset(screen.DATASET)
    meta = ds.meta
    ep_ids = [i for i in range(len(meta.episodes)) if TASK_DESCRIPTION in meta.episodes[i]["tasks"]]
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
        feats = embed_batch(policy, imgs1, imgs2, resize_wh)
        result[split] = {"feats": feats, "states": states, "progress": progress, "ep": ep_of_frame}
        print(f"[{split}] {len(ep_set)} episodes, {len(progress)} frames", flush=True)
    result["n_train_ep"] = len(train_eps)
    result["n_val_ep"] = len(val_eps)
    return result


def train_head(data, epochs=300, lr=1e-3):
    state_mean = data["train"]["states"].mean(axis=0)
    state_std = data["train"]["states"].std(axis=0) + 1e-6

    def make_xy(split):
        feats = data[split]["feats"]
        states = (data[split]["states"] - state_mean) / state_std
        x = np.concatenate([feats, states], axis=1)
        y = data[split]["progress"]
        return torch.tensor(x, dtype=torch.float32, device=DEVICE), torch.tensor(y, dtype=torch.float32, device=DEVICE)

    x_train, y_train = make_xy("train")
    x_val, y_val = make_xy("val")

    head = ProgressHead().to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        pred = head(x_train)
        loss = nn.functional.mse_loss(pred, y_train)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            head.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(head(x_val), y_val).item()
            print(f"  epoch {ep}: train_mse={loss.item():.4f} val_mse={val_loss:.4f}", flush=True)
    train_time = time.time() - t0

    head.eval()
    with torch.no_grad():
        val_pred = head(x_val).cpu().numpy()
    val_true = y_val.cpu().numpy()
    overall_corr = float(np.corrcoef(val_pred, val_true)[0, 1])

    per_ep_corrs = []
    for ep in np.unique(data["val"]["ep"]):
        mask = data["val"]["ep"] == ep
        if mask.sum() >= 3 and val_pred[mask].std() > 1e-8:
            per_ep_corrs.append(float(np.corrcoef(val_pred[mask], val_true[mask])[0, 1]))

    torch.save({"state_dict": head.state_dict(), "state_mean": state_mean, "state_std": state_std}, HEAD_PATH)
    print(f"Saved head to {HEAD_PATH}", flush=True)

    return head, state_mean, state_std, {
        "train_time_s": train_time,
        "n_train_frames": len(y_train),
        "n_val_frames": len(y_val),
        "n_train_episodes": data["n_train_ep"],
        "n_val_episodes": data["n_val_ep"],
        "held_out_overall_pearson_r": overall_corr,
        "held_out_per_episode_pearson_r_mean": float(np.mean(per_ep_corrs)) if per_ep_corrs else None,
        "held_out_per_episode_pearson_r_n": len(per_ep_corrs),
        "held_out_val_mse": float(np.mean((val_pred - val_true) ** 2)),
    }


def predict_progress(head, state_mean, state_std, feats, states):
    states_n = (states - state_mean) / state_std
    x = np.concatenate([feats, states_n], axis=1)
    with torch.no_grad():
        return head(torch.tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy()


@torch.no_grad()
def collect_rollout(policy, preproc, env, task_description, seed, max_steps):
    """Stripped-down copy of smoke_naive_ttt_safety.py's rollout_episode_with_safety /
    train_grpo.py's rollout_episode: identical flow-SDE rollout mechanics, but only
    records raw (pre-preprocessor) image/state per decision point instead of safety
    proxy positions or replayable transitions -- this file trains no policy update."""
    policy.eval()
    policy._queues[ACTION] = deque([], maxlen=policy.config.n_action_steps)
    for key in list(policy._queues.keys()):
        if key != ACTION:
            policy._queues[key] = deque([], maxlen=policy.config.n_action_steps)

    observation, _ = env.reset(seed=seed)
    imgs1, imgs2, states = [], [], []
    success = False
    step = 0
    for step in range(1, max_steps + 1):
        if len(policy._queues[ACTION]) == 0:
            raw = suite_screen._policy_input(observation, task_description)
            imgs1.append(raw["observation.images.image"].squeeze(0).cpu())
            imgs2.append(raw["observation.images.wrist_image"].squeeze(0).cpu())
            states.append(raw["observation.state"].squeeze(0).numpy())

            batch = preproc(raw)
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
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
            noise = policy.model.sample_noise(actions_shape, DEVICE)
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

    return {"success": success, "steps": step, "imgs1": imgs1, "imgs2": imgs2,
            "states": np.stack(states).astype(np.float32)}


def run_rollout_validation(screen, policy, preproc, head, state_mean, state_std):
    env = screen._make_env(TASK_ID)
    resize_wh = policy.config.resize_imgs_with_padding
    per_rollout = []
    for seed in VAL_ROLLOUT_SEEDS:
        out = collect_rollout(policy, preproc, env, TASK_DESCRIPTION, seed, screen.MAX_STEPS)
        feats = embed_batch(policy, out["imgs1"], out["imgs2"], resize_wh)
        pred = predict_progress(head, state_mean, state_std, feats, out["states"])
        n = len(pred)
        t_norm = np.arange(n) / max(n - 1, 1)
        corr_vs_time = float(np.corrcoef(pred, t_norm)[0, 1]) if n >= 3 and pred.std() > 1e-8 else None
        rec = {
            "seed": seed, "orig_label_success": VAL_ROLLOUT_ORIG_LABEL[seed],
            "fresh_success": out["success"], "steps": out["steps"],
            "pred_progress_start": float(pred[0]), "pred_progress_end": float(pred[-1]),
            "pred_progress_delta": float(pred[-1] - pred[0]),
            "pred_progress_mean_first_half": float(pred[: n // 2].mean()) if n >= 2 else float(pred[0]),
            "pred_progress_mean_second_half": float(pred[n // 2:].mean()) if n >= 2 else float(pred[-1]),
            "corr_pred_vs_timestep": corr_vs_time,
        }
        per_rollout.append(rec)
        print(f"[rollout seed={seed}] fresh_success={out['success']} steps={out['steps']} "
              f"pred_progress: start={rec['pred_progress_start']:.3f} end={rec['pred_progress_end']:.3f} "
              f"delta={rec['pred_progress_delta']:+.3f} corr_vs_t={corr_vs_time}", flush=True)

    succ = [r for r in per_rollout if r["fresh_success"]]
    fail = [r for r in per_rollout if not r["fresh_success"]]
    summary = {
        "n_success": len(succ), "n_fail": len(fail),
        "success_mean_delta": float(np.mean([r["pred_progress_delta"] for r in succ])) if succ else None,
        "fail_mean_delta": float(np.mean([r["pred_progress_delta"] for r in fail])) if fail else None,
        "success_mean_end": float(np.mean([r["pred_progress_end"] for r in succ])) if succ else None,
        "fail_mean_end": float(np.mean([r["pred_progress_end"] for r in fail])) if fail else None,
        "fail_end_values": [r["pred_progress_end"] for r in fail],
        "success_end_values": [r["pred_progress_end"] for r in succ],
    }
    return per_rollout, summary


def main():
    t0 = time.time()
    screen, policy, preproc = load_backbone()

    print("=== Collecting demo features (train/val split) ===", flush=True)
    data = collect_demo_dataset(screen, policy)

    print("=== Training progress head ===", flush=True)
    head, state_mean, state_std, train_stats = train_head(data)
    print(f"Held-out overall Pearson r = {train_stats['held_out_overall_pearson_r']:.4f} "
          f"(per-episode mean r = {train_stats['held_out_per_episode_pearson_r_mean']})", flush=True)

    print("=== Rollout validation (fresh episodes, same checkpoint/task/config) ===", flush=True)
    per_rollout, rollout_summary = run_rollout_validation(screen, policy, preproc, head, state_mean, state_std)

    total_time = time.time() - t0
    out = {
        "meta": {
            "suite": SUITE, "task_id": TASK_ID, "task_description": TASK_DESCRIPTION,
            "ckpt": CKPT, "backbone": "policy.model.vlm_with_expert.embed_image (frozen, no_grad)",
            "head_arch": "1928 -> 512 -> 128 -> 1, ReLU, sigmoid output",
            "label": "self-supervised t/(T-1) on LIBERO expert demos",
            "total_wall_time_s": total_time,
        },
        "training": train_stats,
        "rollout_validation": {"per_rollout": per_rollout, "summary": rollout_summary},
    }
    EVAL_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {EVAL_PATH}", flush=True)
    print(f"Total wall time: {total_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
