"""Fresh paired GPU-batched evaluation; see PREREG_BATCHED_TIMING_2026-09-28.md.

Uses the unmodified harness and sampler. Never merges with scalar-run outcomes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
import harness as h


def resample_job(job):
    chunk, arm, hold, k = job
    h.K = k
    return h.apply_arm(chunk, arm, hold)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=["native", "zoh", "qp", "tac_fold", "tac_fold_satfix", "qp_anchor", "spline", "bspline_eps_raw"])
    args = ap.parse_args()
    if args.output.exists():
        ap.error("refusing to overwrite an existing result")
    if args.workers < 1:
        ap.error("workers must be positive")
    root = Path(__file__).resolve().parent
    ref = json.loads(args.reference.read_text())
    assert ref["task"] == args.task and ref["policy"] == "dp" and ref.get("head", "step") == "step"
    checkpoint = root / f"dp_{args.task}.pt"
    assert h.sha256_file(checkpoint) == ref["checkpoint_sha256"], "checkpoint mismatch"
    torch.set_num_threads(1)
    n, k = ref["n"], ref["k"]
    arms = args.arms
    if len(set(arms)) != len(arms) or any(a != "native" and a not in h.resample_math.RESAMPLERS for a in arms):
        ap.error("arms must be distinct registered resamplers or native")
    scalar = h.ManiSkillSim(args.task)
    try:
        _, eps = scalar.split(ref["n_train"], n, offset=ref["eval_offset"])
        states = [scalar._state(ep) for ep in eps]
        cfg, hold = scalar.cfg, scalar.n_hold
    finally:
        scalar.close()
    cache = root / f"demos_{args.task}_{ref['n_train']}.npz"
    z = np.load(cache, allow_pickle=True)
    obs, actions = np.concatenate(z["O"]), np.concatenate(z["A"])
    onorm, anorm = h.MinMax(obs), h.MinMax(actions)
    policy = h.DiffusionPolicy(obs.shape[-1], actions.shape[-1])
    policy.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    policy.cuda().eval()
    stacked = {g: {key: torch.cat([s[g][key] for s in states]) for key in states[0][g]}
               for g in states[0]}
    report = {"task": args.task, "policy": "dp", "k": k, "n": n,
              "eval_offset": ref["eval_offset"], "n_train": ref["n_train"],
              "episode_ids": [ep["episode_id"] for ep in eps],
              "episode_seeds": [ep["episode_seed"] for ep in eps],
              "checkpoint_sha256": h.sha256_file(checkpoint),
              "normalizer_cache_sha256": h.sha256_file(cache),
              "eval_protocol": "paired_batched_gpu_heldout_demo_state",
              "batch_size": n, "arms": arms, "success": {}, "arm_seconds": {},
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "scalar_results_combinable": False, "status": "incomplete"}
    started = time.perf_counter()

    def save():
        report["wall_seconds"] = time.perf_counter() - started
        tmp = args.output.with_suffix(args.output.suffix + ".partial")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output)

    env = gym.make(args.task, num_envs=n, obs_mode="state", control_mode=cfg["ctrl"],
                   sim_backend="physx_cuda", reconfiguration_freq=1)
    try:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
            for arm in arms:
                start = time.perf_counter()
                env.reset(seed=report["episode_seeds"])
                env.unwrapped.set_state_dict(stacked)
                current = h.to_np(env.unwrapped.get_obs()).astype(np.float32)
                previous = current.copy()
                success = np.zeros(n, bool)
                steps = 0
                replan = 0
                while steps < cfg["max_steps"] and not success.all():
                    o = torch.as_tensor(onorm.norm(np.stack([previous, current], axis=1)), device="cuda")
                    noise = []
                    for i in range(n):
                        torch.manual_seed(1_000_003 * i + replan)
                        noise.append(torch.randn(1, policy.horizon, policy.act_dim, device="cuda"))
                    # Inject only the initial noise; run every DDIM operation in the
                    # original sample(). Fail if its random-draw contract ever changes.
                    with patch.object(torch, "randn", return_value=torch.cat(noise)) as draw:
                        pred = policy.sample(o)
                        draw.assert_called_once_with(n, policy.horizon, policy.act_dim, device=o.device)
                    chunks = np.clip(anorm.denorm(pred.cpu().numpy())[:, :8], -1, 1).astype(np.float32)
                    executed = np.stack(list(pool.map(resample_job,
                        ((c, arm, hold, k) for c in chunks), chunksize=16)))
                    assert np.isfinite(executed).all(), "nonfinite resampler output"
                    for t in range(min(8, cfg["max_steps"] - steps)):
                        action = executed[:, t].copy()
                        action[success] = 0
                        out, _, _, _, info = env.step(action)
                        previous, current = current, h.to_np(out).astype(np.float32)
                        success |= h.to_np(info["success"]).reshape(-1).astype(bool)
                        steps += 1
                        if success.all():
                            break
                    replan += 1
                report["success"][arm] = success.tolist()
                report["arm_seconds"][arm] = time.perf_counter() - start
                save()
                print(f"{arm}: {int(success.sum())}/{n}, {report['arm_seconds'][arm]:.1f}s", flush=True)
        report["status"] = "complete"
        save()
    finally:
        env.close()


if __name__ == "__main__":
    main()
