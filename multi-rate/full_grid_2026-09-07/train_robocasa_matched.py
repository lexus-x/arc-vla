#!/usr/bin/env python3
"""Train Diffusion Policy and Regression Policy on RoboCasa (RC-OpenDrawer)
matching the Push-T benchmark budget:
- 537,000 gradient steps
- Regression Policy (ACT-style feedforward baseline)
- Diffusion Policy (DDIM-10 baseline)
- Closed-loop evaluation across 1X, 2X, 4X, 8X speedups
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dp_min import DiffusionPolicy, MinMax, make_chunks, EMA
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


def apply_arm(chunk, arm, n_hold, k):
    if arm == "native":
        return chunk
    nd = chunk.shape[1] - n_hold
    out = decimate_and_resample(chunk[:, :nd], k, arm)
    if n_hold:
        T = len(chunk)
        nb = T // k
        g = np.repeat(chunk[:nb * k:k, nd:], k, 0)
        if nb * k < T:
            g = np.concatenate([g, chunk[nb * k:, nd:]], 0)
        out = np.concatenate([out, g], 1)
    return out.astype(np.float32)


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
                    exec_chunk = apply_arm(np.clip(chunk, -1, 1).astype(np.float32), arm, sim.n_hold, k)

                    for a in exec_chunk:
                        obs, s_t, done = sim.step(a)
                        hist.append(obs)
                        steps += 1
                        s |= s_t
                        if s or done or steps >= sim.max_steps:
                            break
                    replan += 1
                succ[arm].append(s)

            rates = " ".join(f"{a}={sum(succ[a])}/{len(succ[a])}" for a in arms)
            print(f"  [{model_type} k={k}] [{ei+1}/{len(eval_eps)}] {rates}", flush=True)

        dt = time.time() - t0
        print(f"  [{model_type} k={k}] finished in {dt:.1f}s", flush=True)
        results[f"k{k}"] = {
            a: {"success_rate": float(np.mean(succ[a])), "count": f"{sum(succ[a])}/{len(succ[a])}"} for a in arms
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="RC-OpenDrawer", help="RoboCasa task name")
    parser.add_argument("--steps", type=int, default=537_000, help="Number of training steps")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_train", type=int, default=39, help="Number of training demos")
    parser.add_argument("--n_eval", type=int, default=15, help="Number of eval demos")
    parser.add_argument("--port", type=int, default=8765, help="Bridge port")
    parser.add_argument("--models", default="reg,diff", help="Comma-separated models: reg, diff")
    parser.add_argument("--skip_eval", action="store_true", help="Skip closed-loop evaluation")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (100 steps, 2 eval eps)")
    args = parser.parse_args()

    if args.smoke:
        args.steps = 100
        args.n_eval = 2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting RoboCasa ({args.task}) {args.steps} Steps Training & Evaluation on {dev} ===", flush=True)

    cache = f"{HERE}/demos_{args.task}_{args.n_train}.npz"
    if not os.path.exists(cache):
        raise FileNotFoundError(f"Demonstration cache {cache} not found!")

    z = np.load(cache, allow_pickle=True)
    O, A = list(z["O"]), list(z["A"])
    print(f"[data] loaded {cache} ({len(O)} demos)", flush=True)

    onorm, anorm = MinMax(np.concatenate(O)), MinMax(np.concatenate(A))
    chunks = [make_chunks(onorm.norm(o), anorm.norm(a), 2, 16) for o, a in zip(O, A)]
    OC, AC = np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])
    obs_dim = OC.shape[-1]
    act_dim = AC.shape[-1]
    print(f"[data] {len(OC)} windows, obs_dim={obs_dim}, act_dim={act_dim}", flush=True)

    models_to_run = [m.strip().lower() for m in args.models.split(",")]
    final_results = {}

    # 1. REGRESSION POLICY
    if "reg" in models_to_run:
        reg_model = RegressionPolicy(obs_dim, act_dim, horizon=16)
        reg_ema = train_model(
            reg_model, OC, AC, steps=args.steps, bs=args.bs, lr=args.lr, dev=dev, name="Reg. Policy",
            log_every=50 if args.smoke else 25000
        )
        reg_ckpt = f"{HERE}/reg_{args.task}_{args.steps // 1000}k.pt" if not args.smoke else f"{HERE}/reg_{args.task}_smoke.pt"
        torch.save(reg_ema.state_dict(), reg_ckpt)
        print(f"[saved] {reg_ckpt}", flush=True)

        if not args.skip_eval:
            try:
                from harness import RoboCasaSim
                sim = RoboCasaSim(args.task, port=args.port)
                eval_eps = list(range(args.n_train, args.n_train + args.n_eval))
                reg_results = evaluate_policy(reg_ema, "Reg", sim, eval_eps, onorm, anorm, ks=[1, 2, 4, 8], dev=dev)
                final_results["Reg"] = reg_results
                sim.close()
            except Exception as e:
                print(f"[eval warning] Could not run closed-loop evaluation: {e}", flush=True)

    # 2. DIFFUSION POLICY
    if "diff" in models_to_run:
        diff_model = DiffusionPolicy(obs_dim, act_dim, horizon=16)
        diff_ema = train_model(
            diff_model, OC, AC, steps=args.steps, bs=args.bs, lr=args.lr, dev=dev, name="Diff. Policy",
            log_every=50 if args.smoke else 25000
        )
        diff_ckpt = f"{HERE}/dp_{args.task}_{args.steps // 1000}k.pt" if not args.smoke else f"{HERE}/dp_{args.task}_smoke.pt"
        torch.save(diff_ema.state_dict(), diff_ckpt)
        print(f"[saved] {diff_ckpt}", flush=True)

        if not args.skip_eval:
            try:
                from harness import RoboCasaSim
                sim = RoboCasaSim(args.task, port=args.port)
                eval_eps = list(range(args.n_train, args.n_train + args.n_eval))
                diff_results = evaluate_policy(diff_ema, "Diff", sim, eval_eps, onorm, anorm, ks=[1, 2, 4, 8], dev=dev)
                final_results["Diff"] = diff_results
                sim.close()
            except Exception as e:
                print(f"[eval warning] Could not run closed-loop evaluation: {e}", flush=True)

    if final_results:
        out_file = f"{HERE}/result_rc_{args.task.lower().replace('rc-', '')}_matched_{args.steps // 1000}k.json"
        with open(out_file, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"[DONE] Benchmark results saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
