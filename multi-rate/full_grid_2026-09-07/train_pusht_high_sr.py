#!/usr/bin/env python3
"""Train Diffusion Policy on 600 Push-T demonstrations with calibrated normalization.
Saves checkpoints and evaluates diagnostics on held-out test episodes.
"""
import argparse, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dp_min import DiffusionPolicy, MinMax, make_chunks, EMA
from harness import ManiSkillSim
from eval_pusht_diagnostics import evaluate_diagnostics


def train_policy(policy, obs_chunks, act_chunks, steps=100_000, bs=256, lr=1e-4, dev="cuda",
                 snap_every=25_000, snap_prefix="dp_PushT_600"):
    policy.to(dev).train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    ema = EMA(policy)
    O = torch.as_tensor(obs_chunks, device=dev)
    A = torch.as_tensor(act_chunks, device=dev)
    N = len(O)

    t0 = time.time()
    print(f"\n[train] starting {steps} steps (batch_size={bs}, lr={lr}, data_windows={N})...", flush=True)
    for s in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), device=dev)
        l = policy.loss(O[idx], A[idx])
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        sched.step()
        ema.update(policy)

        if s % 5000 == 0 or s == 1 or s == steps:
            dt = time.time() - t0
            steps_per_sec = s / dt
            eta_min = (steps - s) / steps_per_sec / 60
            print(f"[train] step {s}/{steps} | loss={l.item():.5f} | {steps_per_sec:.1f} steps/s | ETA: {eta_min:.1f} min", flush=True)

        if s % snap_every == 0 or s == steps:
            ckpt_path = f"{HERE}/{snap_prefix}_step{s}.pt"
            latest_path = f"{HERE}/{snap_prefix}_latest.pt"
            torch.save(ema.m.state_dict(), ckpt_path)
            torch.save(ema.m.state_dict(), latest_path)
            print(f"[train] snapshot saved -> {latest_path} (step {s})", flush=True)

    dt_total = time.time() - t0
    print(f"[train] completed {steps} steps in {dt_total/60:.1f} min ({steps/dt_total:.1f} steps/s).", flush=True)
    return ema.m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data", type=str, default=f"{HERE}/demos_PushT-v1_600.npz")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_start", type=int, default=600)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Push-T High-SR Training on {dev} ===")
    print(f"Dataset: {args.data} | Steps: {args.steps} | Batch size: {args.batch_size}")

    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found. Run harvest_pusht_600.py first!")
        sys.exit(1)

    z = np.load(args.data, allow_pickle=True)
    O, A = list(z["O"]), list(z["A"])
    print(f"[data] loaded {len(O)} demonstrations")

    onorm = MinMax(np.concatenate(O))
    anorm = MinMax(np.concatenate(A))
    norm_path = f"{HERE}/dp_PushT_600_norm.pt"
    torch.save({"onorm": onorm.state_dict(), "anorm": anorm.state_dict()}, norm_path)
    print(f"[norm] saved normalization parameters -> {norm_path}")

    chunks = [make_chunks(onorm.norm(o), anorm.norm(a), 2, 16) for o, a in zip(O, A)]
    OC = np.concatenate([c[0] for c in chunks])
    AC = np.concatenate([c[1] for c in chunks])
    obs_dim, act_dim = OC.shape[-1], AC.shape[-1]
    print(f"[data] generated {len(OC)} training windows (obs_dim={obs_dim}, act_dim={act_dim})")

    model = DiffusionPolicy(obs_dim, act_dim, horizon=16)
    ema_model = train_policy(model, OC, AC, steps=args.steps, bs=args.batch_size, lr=args.lr, dev=dev)

    # Save final model
    final_path = f"{HERE}/dp_PushT_600_final.pt"
    torch.save(ema_model.state_dict(), final_path)
    print(f"[saved] final model -> {final_path}")

    # Evaluate on held-out episodes
    print(f"\n=== Evaluating Diagnostics on {args.eval_episodes} Held-Out Episodes ===")
    sim = ManiSkillSim("PushT-v1")
    eval_eps = sim.eligible[args.eval_start : args.eval_start + args.eval_episodes]

    for n_exec in [8, 4]:
        for max_steps in [150, 200]:
            label = f"exec={n_exec}, max_steps={max_steps}"
            t0 = time.time()
            res = evaluate_diagnostics(
                ema_model, onorm, anorm, sim, eval_eps,
                n_exec=n_exec, max_steps=max_steps, dev=dev
            )
            print(f"[{label}] wall: {time.time()-t0:.1f}s | "
                  f"Coverage: {res['mean_max_coverage']*100:.1f}% | "
                  f"SR(>=0.90): {res['succ_rate_90']*100:.1f}% ({res['count_90']}) | "
                  f"SR(>=0.80): {res['succ_rate_80']*100:.1f}% ({res['count_80']})")

    sim.close()


if __name__ == "__main__":
    main()
