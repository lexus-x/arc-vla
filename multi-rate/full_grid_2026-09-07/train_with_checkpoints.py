#!/usr/bin/env python3
"""Unified training runner for Diffusion Policy and Flow Matching
with periodic snapshot checkpoints for live 30-minute evaluations.
"""
import argparse, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dp_min import DiffusionPolicy, MinMax, make_chunks, EMA
from fm_min import FlowMatchingPolicy


def train_policy(policy, obs_chunks, act_chunks, steps=537_000, bs=256, lr=1e-4, dev="cuda",
                 name="policy", log_every=5000, snap_every=25000, snap_prefix="policy"):
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
    print(f"\n[{name}] starting {steps} steps (batch_size={bs}, lr={lr})...", flush=True)
    for s in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), device=dev)
        l = policy.loss(O[idx], A[idx])
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        sched.step()
        ema.update(policy)

        if s % log_every == 0 or s == 1 or s == steps:
            dt = time.time() - t0
            steps_per_sec = s / dt
            eta_min = (steps - s) / steps_per_sec / 60
            print(f"[{name}] step {s}/{steps} | loss={l.item():.5f} | {steps_per_sec:.1f} steps/s | ETA: {eta_min:.1f} min", flush=True)

        if s % snap_every == 0 or s == steps:
            latest_ckpt = f"{snap_prefix}_latest.pt"
            torch.save(ema.m.state_dict(), latest_ckpt)
            # Record metadata
            meta_file = f"{snap_prefix}_meta.txt"
            with open(meta_file, "w") as f:
                f.write(f"step={s}\nsteps_total={steps}\nloss={l.item():.5f}\ntimestamp={time.time()}\n")
            print(f"[{name}] saved snapshot checkpoint -> {latest_ckpt} (step {s})", flush=True)

    dt_total = time.time() - t0
    print(f"[{name}] finished {steps} steps in {dt_total/60:.2f} minutes ({steps/dt_total:.1f} steps/s).", flush=True)
    return ema.m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task name, e.g. PushT-v1, RC-CloseSingleDoor")
    parser.add_argument("--policy", choices=["dp", "fm"], default="dp", help="Policy type")
    parser.add_argument("--demos", required=True, help="Path to demo npz file")
    parser.add_argument("--steps", type=int, default=537_000, help="Training steps")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--snap_every", type=int, default=25000, help="Snapshot interval")
    parser.add_argument("--out_prefix", default="", help="Prefix for checkpoints")
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = args.out_prefix or f"{HERE}/{args.policy}_{args.task}"
    print(f"=== Training {args.policy.upper()} on {args.task} for {args.steps} steps ===", flush=True)

    z = np.load(args.demos, allow_pickle=True)
    O, A = list(z["O"]), list(z["A"])
    print(f"[data] Loaded {len(O)} demos from {args.demos}", flush=True)

    onorm, anorm = MinMax(np.concatenate(O)), MinMax(np.concatenate(A))
    # Save normalizer state for evaluation
    norm_file = f"{prefix}_norm.pt"
    torch.save({"onorm": onorm.state(), "anorm": anorm.state()}, norm_file)

    chunks = [make_chunks(onorm.norm(o), anorm.norm(a), 2, 16) for o, a in zip(O, A)]
    OC, AC = np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])
    obs_dim = OC.shape[-1]
    act_dim = AC.shape[-1]
    print(f"[data] {len(OC)} windows, obs_dim={obs_dim}, act_dim={act_dim}", flush=True)

    if args.policy == "dp":
        model = DiffusionPolicy(obs_dim, act_dim, horizon=16)
    else:
        model = FlowMatchingPolicy(obs_dim, act_dim, horizon=16)

    trained_model = train_policy(
        model, OC, AC, steps=args.steps, bs=args.bs, lr=args.lr, dev=dev,
        name=f"{args.policy.upper()}-{args.task}", snap_every=args.snap_every, snap_prefix=prefix
    )

    final_ckpt = f"{prefix}_final.pt"
    torch.save(trained_model.state_dict(), final_ckpt)
    print(f"[DONE] Final model saved to {final_ckpt}", flush=True)


if __name__ == "__main__":
    main()
