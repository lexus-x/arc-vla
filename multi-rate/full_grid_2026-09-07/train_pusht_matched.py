#!/usr/bin/env python3
"""Train Diffusion Policy and Regression Policy on Push-T matching the B-spline paper budget:
- 537,000 gradient steps (3,120 epochs equivalent)
- Diffusion Policy (Diff.)
- Regression Policy (Reg.)
- Closed-loop evaluation across 1X, 2X, 4X, 8X speedups.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dp_min import DiffusionPolicy, MinMax, make_chunks, EMA
from harness import ManiSkillSim
from resample_math import decimate_and_resample


class RegressionPolicy(nn.Module):
    """Direct action-chunk regression model (ACT-style feedforward baseline)."""
    def __init__(self, obs_dim, act_dim, n_obs=2, horizon=16, hidden_dim=512):
        super().__init__()
        self.horizon, self.act_dim = horizon, act_dim
        in_dim = obs_dim * n_obs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, horizon * act_dim),
        )

    def forward(self, obs):
        B = obs.shape[0]
        x = obs.flatten(1)
        return self.net(x).view(B, self.horizon, self.act_dim)

    def loss(self, obs, act):
        pred = self.forward(obs)
        return F.l1_loss(pred, act)


def train_model(model, obs_chunks, act_chunks, steps=537_000, bs=256, lr=1e-4, dev="cuda", name="model", log_every=25000):
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    ema = EMA(model)
    O = torch.as_tensor(obs_chunks, device=dev)
    A = torch.as_tensor(act_chunks, device=dev)
    N = len(O)

    t0 = time.time()
    print(f"\n[{name}] starting {steps} steps (batch_size={bs}, lr={lr})...", flush=True)
    for s in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), device=dev)
        l = model.loss(O[idx], A[idx])
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        sched.step()
        ema.update(model)

        if s % log_every == 0 or s == 1 or s == steps:
            dt = time.time() - t0
            steps_per_sec = s / dt
            eta_min = (steps - s) / steps_per_sec / 60
            print(f"[{name}] step {s}/{steps} | loss={l.item():.5f} | {steps_per_sec:.1f} steps/s | ETA: {eta_min:.1f} min", flush=True)

    dt_total = time.time() - t0
    print(f"[{name}] finished {steps} steps in {dt_total/60:.2f} minutes ({steps/dt_total:.1f} steps/s).", flush=True)
    return ema.m


def evaluate_policy(policy, model_type, sim, eval_eps, onorm, anorm, ks=[1, 2, 4, 8], dev="cuda"):
    policy.eval()
    results = {}
    arms = ["native", "zoh"]

    for k in ks:
        print(f"\n[{model_type}] evaluating {k}X (n={len(eval_eps)})...", flush=True)
        succ = {a: [] for a in arms}
        t0 = time.time()
        for ei, ep in enumerate(eval_eps):
            for arm in arms:
                obs = sim.reset_to(ep)
                hist = [obs, obs]
                s = False
                steps = 0
                replan = 0
                while steps < sim.max_steps and not s:
                    o = torch.as_tensor(onorm.norm(np.stack(hist[-2:]))[None], device=dev)
                    if model_type == "Diff":
                        torch.manual_seed(1_000_003 * ei + replan)
                        pred = anorm.denorm(policy.sample(o).cpu().numpy()[0])
                    else:
                        with torch.no_grad():
                            pred = anorm.denorm(policy(o).cpu().numpy()[0])

                    chunk = pred[:8]
                    if arm == "native":
                        exec_chunk = np.clip(chunk, -1, 1).astype(np.float32)
                    else:
                        nd = chunk.shape[1] - sim.n_hold
                        c_res = decimate_and_resample(chunk[:, :nd], k, arm)
                        exec_chunk = np.clip(c_res, -1, 1).astype(np.float32)

                    for a in exec_chunk:
                        obs, s_t, done = sim.step(a)
                        hist.append(obs)
                        steps += 1
                        s |= s_t
                        if s or done or steps >= sim.max_steps:
                            break
                    replan += 1
                succ[arm].append(s)

            if (ei + 1) % 25 == 0 or ei + 1 == len(eval_eps):
                rates = " ".join(f"{a}={sum(succ[a])}/{len(succ[a])}" for a in arms)
                print(f"  [{model_type} k={k}] [{ei+1}/{len(eval_eps)}] {rates}", flush=True)

        dt = time.time() - t0
        print(f"  [{model_type} k={k}] finished in {dt:.1f}s", flush=True)
        results[f"k{k}"] = {
            a: {"success_rate": float(np.mean(succ[a])), "count": f"{sum(succ[a])}/{len(succ[a])}"} for a in arms
        }

    return results


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Push-T 537k Training & Evaluation on {dev} ===", flush=True)

    sim = ManiSkillSim("PushT-v1")
    train_eps, eval_eps = sim.split(200, 100)

    cache = f"{HERE}/demos_PushT-v1_200.npz"
    z = np.load(cache, allow_pickle=True)
    O, A = list(z["O"]), list(z["A"])
    print(f"[data] loaded {cache} ({len(O)} demos)", flush=True)

    onorm, anorm = MinMax(np.concatenate(O)), MinMax(np.concatenate(A))
    chunks = [make_chunks(onorm.norm(o), anorm.norm(a), 2, 16) for o, a in zip(O, A)]
    OC, AC = np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])
    obs_dim = OC.shape[-1]
    act_dim = AC.shape[-1]
    print(f"[data] {len(OC)} windows, obs_dim={obs_dim}, act_dim={act_dim}", flush=True)

    final_results = {}

    # 1. TRAIN & EVALUATE DIFFUSION POLICY (537,000 STEPS)
    diff_model = DiffusionPolicy(obs_dim, act_dim, horizon=16)
    diff_ema = train_model(diff_model, OC, AC, steps=537_000, bs=256, lr=1e-4, dev=dev, name="Diff. Policy", log_every=25000)
    torch.save(diff_ema.state_dict(), f"{HERE}/dp_PushT-v1_537k.pt")
    print(f"[saved] {HERE}/dp_PushT-v1_537k.pt", flush=True)
    diff_results = evaluate_policy(diff_ema, "Diff", sim, eval_eps, onorm, anorm, ks=[1, 2, 4, 8], dev=dev)
    final_results["Diff"] = diff_results

    # 2. TRAIN & EVALUATE REGRESSION POLICY (537,000 STEPS)
    reg_model = RegressionPolicy(obs_dim, act_dim, horizon=16)
    reg_ema = train_model(reg_model, OC, AC, steps=537_000, bs=256, lr=1e-4, dev=dev, name="Reg. Policy", log_every=25000)
    torch.save(reg_ema.state_dict(), f"{HERE}/reg_PushT-v1_537k.pt")
    print(f"[saved] {HERE}/reg_PushT-v1_537k.pt", flush=True)
    reg_results = evaluate_policy(reg_ema, "Reg", sim, eval_eps, onorm, anorm, ks=[1, 2, 4, 8], dev=dev)
    final_results["Reg"] = reg_results

    sim.close()

    out_file = f"{HERE}/result_pusht_matched_537k.json"
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[DONE] Final Push-T Matched Benchmark saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
