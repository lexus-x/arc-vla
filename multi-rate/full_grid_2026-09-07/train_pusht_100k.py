#!/usr/bin/env python3
"""Train Diffusion Policy on Push-T for 100,000 steps (matching the full training budget) and evaluate 1X, 2X, 4X, 8X."""
import json, math, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dp_min import DiffusionPolicy, MinMax, make_chunks, train
from harness import ManiSkillSim, apply_arm, exact_mcnemar

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Training Push-T Diffusion Policy for 100,000 steps ===", flush=True)

    sim = ManiSkillSim("PushT-v1")
    train_eps, eval_eps = sim.split(200, 100)

    cache = f"{HERE}/demos_PushT-v1_200.npz"
    z = np.load(cache, allow_pickle=True)
    O, A = list(z["O"]), list(z["A"])
    print(f"[data] loaded {cache} ({len(O)} demos)", flush=True)

    onorm, anorm = MinMax(np.concatenate(O)), MinMax(np.concatenate(A))
    chunks = [make_chunks(onorm.norm(o), anorm.norm(a), 2, 16) for o, a in zip(O, A)]
    OC, AC = np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])
    print(f"[data] {len(OC)} windows, obs_dim={OC.shape[-1]}, act_dim={AC.shape[-1]}", flush=True)

    ckpt = f"{HERE}/dp_PushT-v1_100k.pt"
    policy = DiffusionPolicy(OC.shape[-1], AC.shape[-1], horizon=16)

    t0_train = time.time()
    print(f"[train] starting 100,000 steps on {dev}...", flush=True)
    policy = train(policy, OC, AC, steps=100_000, bs=256, lr=1e-4, dev=dev, log_every=5000)
    torch.save(policy.state_dict(), ckpt)
    dt_train = time.time() - t0_train
    print(f"[train] saved {ckpt} in {dt_train/60:.2f} minutes", flush=True)
    policy.eval()

    # Evaluate across 1X, 2X, 4X, 8X
    ks = [1, 2, 4, 8]
    arms = ["native", "zoh"]
    all_results = {}

    for k in ks:
        print(f"\n--- Evaluating Push-T 100k model at {k}X (n={len(eval_eps)}) ---", flush=True)
        succ = {a: [] for a in arms}
        t0_eval = time.time()
        for ei, ep in enumerate(eval_eps):
            for arm in arms:
                obs = sim.reset_to(ep)
                hist = [obs, obs]
                s = False
                steps = 0
                replan = 0
                while steps < sim.max_steps and not s:
                    o = torch.as_tensor(onorm.norm(np.stack(hist[-2:]))[None], device=dev)
                    torch.manual_seed(1_000_003 * ei + replan)
                    pred = anorm.denorm(policy.sample(o).cpu().numpy()[0])
                    chunk = pred[:8]
                    if arm == "native":
                        exec_chunk = np.clip(chunk, -1, 1).astype(np.float32)
                    else:
                        from resample_math import decimate_and_resample
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

            if (ei + 1) % 20 == 0 or ei + 1 == len(eval_eps):
                rates = " ".join(f"{a}={sum(succ[a])}/{len(succ[a])}" for a in arms)
                print(f"  [k={k}] [{ei+1}/{len(eval_eps)}] {rates}", flush=True)

        print(f"  [k={k}] completed in {time.time()-t0_eval:.1f}s", flush=True)
        all_results[f"k{k}"] = {a: {"success_rate": float(np.mean(succ[a])), "count": f"{sum(succ[a])}/{len(succ[a])}"} for a in arms}

    sim.close()
    out_file = f"{HERE}/result_dp_PushT-v1_100k_grid.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Results saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
