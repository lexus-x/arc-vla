#!/usr/bin/env python3
"""Harvest 600 demonstrations from pusht_rl.h5 with calibrated action bounds.
Saves to demos_PushT-v1_600.npz.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from harness import ManiSkillSim

OUT_FILE = f"{HERE}/demos_PushT-v1_600.npz"


def main():
    sim = ManiSkillSim("PushT-v1")
    n_demos = 600
    eps = sim.eligible[:n_demos]
    print(f"Harvesting {len(eps)} demonstrations from pusht_rl...")

    t0 = time.time()
    O, A = [], []
    for i, ep in enumerate(eps):
        acts = np.asarray(sim.h5[f"traj_{int(ep['episode_id'])}"]["actions"], np.float32)
        # Clamped action trajectory matching controller physical bounds
        acts_clamped = np.clip(acts, -1.0, 1.0)
        obs = sim.reset_to(ep)
        ob = []
        for t in range(len(acts_clamped)):
            ob.append(obs)
            obs, _, done = sim.step(acts_clamped[t])
            if done:
                break
        n = len(ob)
        O.append(np.stack(ob))
        A.append(acts_clamped[:n])
        if (i + 1) % 50 == 0 or (i + 1) == len(eps):
            dt = time.time() - t0
            print(f"[{i+1}/{len(eps)}] harvested in {dt:.1f}s ({dt/(i+1):.2f}s/demo)", flush=True)

    sim.close()
    np.savez_compressed(OUT_FILE, O=np.array(O, dtype=object), A=np.array(A, dtype=object))
    print(f"Saved {len(O)} demonstrations to {OUT_FILE} (total time: {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
