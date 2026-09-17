#!/usr/bin/env python3
"""Diagnostic evaluation for Push-T:
Measures binary success at 90% threshold, 80% threshold, and mean max coverage score (IoU).
Tests execution chunk size (8 vs 4) and max step horizons (150 vs 200).
"""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dp_min import DiffusionPolicy, MinMax
from harness import ManiSkillSim


def evaluate_diagnostics(model, onorm, anorm, sim, eval_eps, n_exec=8, max_steps=150, ddim_steps=10, dev="cuda"):
    model.eval()
    coverages = []
    succ_90 = []
    succ_80 = []
    steps_taken = []

    for ei, ep in enumerate(eval_eps):
        obs = sim.reset_to(ep)
        hist = [obs, obs]
        s90 = False
        s80 = False
        max_cov = 0.0
        step = 0
        replan = 0

        while step < max_steps and not s90:
            o = torch.as_tensor(onorm.norm(np.stack(hist[-2:]))[None], device=dev)
            torch.manual_seed(1_000_003 * ei + replan)
            pred = anorm.denorm(model.sample(o, n_steps=ddim_steps).cpu().numpy()[0])
            chunk = np.clip(pred[:n_exec], -1, 1).astype(np.float32)

            for a in chunk:
                obs, s_t, done = sim.step(a)
                cov = float(sim.u.pseudo_render_intersection().cpu().item())
                max_cov = max(max_cov, cov)
                hist.append(obs)
                step += 1

                if max_cov >= 0.90:
                    s90 = True
                if max_cov >= 0.80:
                    s80 = True

                if s90 or done or step >= max_steps:
                    break
            replan += 1

        coverages.append(max_cov)
        succ_90.append(s90)
        succ_80.append(s80)
        steps_taken.append(step)

    return {
        "n": len(eval_eps),
        "n_exec": n_exec,
        "max_steps": max_steps,
        "mean_max_coverage": float(np.mean(coverages)),
        "succ_rate_90": float(np.mean(succ_90)),
        "succ_rate_80": float(np.mean(succ_80)),
        "mean_steps": float(np.mean(steps_taken)),
        "count_90": f"{sum(succ_90)}/{len(succ_90)}",
        "count_80": f"{sum(succ_80)}/{len(succ_80)}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=f"{HERE}/dp_PushT-v1_final.pt")
    parser.add_argument("--norm", type=str, default=f"{HERE}/dp_PushT-v1_norm.pt")
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--eval_start", type=int, default=600)
    parser.add_argument("--ddim_steps", type=int, default=10)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.ckpt}")
    sim = ManiSkillSim("PushT-v1")
    eval_eps = sim.eligible[args.eval_start : args.eval_start + args.n_eval]

    norms = torch.load(args.norm, map_location="cpu", weights_only=False)
    onorm = MinMax.from_state(norms["onorm"])
    anorm = MinMax.from_state(norms["anorm"])

    obs0 = sim.reset_to(eval_eps[0])
    model = DiffusionPolicy(len(obs0), 6, horizon=16)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False))
    model.to(dev).eval()

    configs = [
        {"n_exec": 8, "max_steps": 150, "label": "Baseline (exec=8, max=150)"},
        {"n_exec": 4, "max_steps": 150, "label": "Receding Horizon (exec=4, max=150)"},
        {"n_exec": 4, "max_steps": 200, "label": "Extended Budget (exec=4, max=200)"},
    ]

    print(f"\nEvaluating on {len(eval_eps)} held-out episodes [index {args.eval_start}:{args.eval_start+args.n_eval}]...")
    for cfg in configs:
        t0 = time.time()
        res = evaluate_diagnostics(
            model, onorm, anorm, sim, eval_eps,
            n_exec=cfg["n_exec"], max_steps=cfg["max_steps"],
            ddim_steps=args.ddim_steps, dev=dev
        )
        dt = time.time() - t0
        print(f"\n--- {cfg['label']} (wall: {dt:.1f}s) ---")
        print(f"  Mean Max Coverage (Paper IoU): {res['mean_max_coverage']*100:.1f}%")
        print(f"  Binary Success (>=0.90):       {res['succ_rate_90']*100:.1f}% ({res['count_90']})")
        print(f"  Binary Success (>=0.80):       {res['succ_rate_80']*100:.1f}% ({res['count_80']})")
        print(f"  Mean Steps:                    {res['mean_steps']:.1f}")

    sim.close()


if __name__ == "__main__":
    main()
