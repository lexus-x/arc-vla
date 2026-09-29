"""Probe native ManiSkill batching; reports drift, never treats it as SR data."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
import gymnasium as gym
import harness as h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.batch < 2:
        ap.error("batch must be at least two")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    sim = h.ManiSkillSim(args.task)
    _, eps = sim.split(200, args.batch)
    states = [sim._state(ep) for ep in eps]
    actions = [np.asarray(sim.h5[f"traj_{int(ep['episode_id'])}"]["actions"], np.float32)[:16]
               for ep in eps]
    assert all(len(a) == 16 for a in actions)
    report = {"kind": "vector_throughput_probe_not_SR", "task": args.task,
              "batch": args.batch, "steps": 16}
    refs = []
    start = time.perf_counter()
    try:
        for i in range(2):
            trace = [sim.reset_to(eps[i])]
            for action in actions[i]:
                trace.append(sim.step(np.clip(action, -1, 1))[0])
            refs.append(np.stack(trace))
    finally:
        sim.close()
    report["scalar_two_episodes_seconds"] = time.perf_counter() - start
    cfg = h.MANISKILL[args.task]
    start = time.perf_counter()
    env = gym.make(args.task, num_envs=args.batch, obs_mode="state",
                   control_mode=cfg["ctrl"], sim_backend="physx_cuda", reconfiguration_freq=1)
    try:
        env.reset(seed=[ep["episode_seed"] for ep in eps])
        stacked = {g: {n: torch.cat([s[g][n] for s in states]) for n in states[0][g]}
                   for g in states[0]}
        env.unwrapped.set_state_dict(stacked)
        trace = [h.to_np(env.unwrapped.get_obs())]
        for t in range(16):
            obs, _, _, _, _ = env.step(np.clip(np.stack([a[t] for a in actions]), -1, 1))
            trace.append(h.to_np(obs))
        batched = np.stack(trace)
    finally:
        env.close()
    report["batched_episodes_seconds_including_setup"] = time.perf_counter() - start
    report["physics_max_abs_error"] = [float(np.max(np.abs(refs[i] - batched[:, i]))) for i in range(2)]
    report["physics_exact"] = [bool(np.array_equal(refs[i], batched[:, i])) for i in range(2)]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)

    z = np.load(root / f"demos_{args.task}_200.npz", allow_pickle=True)
    obs, acts = np.concatenate(z["O"]), np.concatenate(z["A"])
    norm = h.MinMax(obs)
    policy = h.DiffusionPolicy(obs.shape[-1], acts.shape[-1])
    policy.load_state_dict(torch.load(root / f"dp_{args.task}.pt", weights_only=True, map_location="cpu"))
    policy.cuda().eval()
    inputs = torch.as_tensor(norm.norm(obs[:args.batch * 2].reshape(args.batch, 2, -1)), device="cuda")
    noise = []
    for i in range(args.batch):
        torch.manual_seed(1_000_003 * i)
        noise.append(torch.randn(1, policy.horizon, policy.act_dim, device="cuda"))
    with patch.object(torch, "randn", return_value=torch.cat(noise)):
        policy.sample(inputs)  # warm up this shape
        torch.cuda.synchronize()
        start = time.perf_counter()
        pred = policy.sample(inputs)
        torch.cuda.synchronize()
        report["batch_sample_seconds"] = time.perf_counter() - start
    report["sample_max_abs_error"] = []
    for i in range(2):
        with patch.object(torch, "randn", return_value=noise[i]):
            ref = policy.sample(inputs[i:i + 1])
        report["sample_max_abs_error"].append((ref - pred[i:i + 1]).abs().max().item())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
