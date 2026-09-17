"""Decoder replay: folding vs spline on PickCube-v1 RL demos, paired, in-sim.

Each 50-step demo is decimated to 25 coarse deltas (summing pairs), then reconstructed to 50:
  folding : split each coarse delta in half  -> preserves each cell integral EXACTLY
  spline  : CubicSpline to 50 * (25/50)      -> the campaign resampler, 11.67% displacement err
Arm A replays the untouched actions as a sanity control (must reproduce the demo success rate).
Pre-registered: folding > spline, exact McNemar p<0.05, or the content is chatter and item 1 closes.
"""
import json, os, sys, numpy as np, h5py, torch, gymnasium as gym
from scipy.interpolate import CubicSpline
import mani_skill.envs  # noqa

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 150
D = os.path.expanduser("~/maniskill_data")
meta = json.load(open(f"{D}/pick_rl_joint.json"))
eps = meta["episodes"]

def fold(coarse, n):                       # exact cell integral -> equal split
    k = n // coarse.shape[0]
    return np.repeat(coarse / k, k, axis=0)

def spline(coarse, n):
    x = np.linspace(0, 1, coarse.shape[0])
    return CubicSpline(x, coarse, axis=0)(np.linspace(0, 1, n)) * coarse.shape[0] / n

env = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
               control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")

res = {"orig": [], "fold": [], "spline": []}
with h5py.File(f"{D}/pick_rl_joint.h5", "r") as h:
    for e in eps[:N_EP]:
        a = np.array(h[f"traj_{e['episode_id']}"]["actions"]).astype(np.float32)
        n = (len(a)//2)*2; a = a[:n]
        coarse = a.reshape(n//2, 2, a.shape[1]).sum(1)
        arms = {"orig": a, "fold": fold(coarse, n).astype(np.float32),
                "spline": spline(coarse, n).astype(np.float32)}
        g = h[f"traj_{e['episode_id']}"]["env_states"]
        st = {grp: {nm: torch.as_tensor(np.array(g[grp][nm])[0:1])
                    for nm in g[grp]} for grp in g}
        for k, acts in arms.items():
            env.reset(seed=e["episode_seed"])
            env.unwrapped.set_state_dict(st)
            ok = False
            for t in range(len(acts)):
                _, _, term, trunc, info = env.step(acts[t][None])
                s = info.get("success")
                if s is not None and bool(np.asarray(s).reshape(-1)[0]): ok = True
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
            res[k].append(ok)
env.close()

for k, v in res.items():
    print(f"{k:8s} {sum(v):>4d}/{len(v):<4d} = {100*np.mean(v):>5.1f}%")

f, s = np.array(res["fold"]), np.array(res["spline"])
b = int((f & ~s).sum()); c = int((~f & s).sum())
from math import comb
n2 = b + c
p = 1.0 if n2 == 0 else min(1.0, 2*sum(comb(n2,i) for i in range(min(b,c)+1))/2**n2)
print(f"\nfold-spline delta = {100*(f.mean()-s.mean()):+.1f} pp   discordant {b}/{c}   exact McNemar p={p:.4f}")
