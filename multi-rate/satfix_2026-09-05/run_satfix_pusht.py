"""
satfix on ManiSkill PushT-v1, open-loop demo replay -- same methodology/arms as
run_satfix_general.py (PickCube/LiftPegUpright/PushCube), extended to a task not in that
script's TASK_CONFIGS. control_mode=pd_ee_delta_pose, has_gripper=False (pusher has no
gripper). Confirmed live env.action_space is Box(-1,1,(6,)) despite recorded RL-policy
actions in pusht_rl.h5 exceeding that range unclipped (pre-clip logged network outputs) --
i.e. this task's raw demos saturate heavily under replay, a strong test case for satfix.

Usage: python run_satfix_pusht.py [n_episodes] [k]
"""
import sys, json, numpy as np, h5py, os
sys.path.insert(0, "/home/user/Desktop/multi-rate/vla-vault/scratch")
import gymnasium as gym, mani_skill.envs  # noqa
from sweep_maniskill_decimation_ratios import (resample_exact_integral,
    resample_cubic_spline, resample_pchip, resample_tac_fold, exact_mcnemar, paired_bootstrap_ci)

SD = "/tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
H5 = "/home/user/maniskill_data/pusht_rl.h5"
JS = "/home/user/maniskill_data/pusht_rl.json"


def satfix(pose, delta, K):
    D = delta.shape[1]; out = pose.copy(); n = delta.shape[0]
    for i in range(n):
        v = out[i*K:(i+1)*K, :D].copy(); S = delta[i]
        for _ in range(K + 2):
            v = np.clip(v, -1.0, 1.0); deficit = S - v.sum(0)
            if np.all(np.abs(deficit) < 1e-7): break
            can = np.where(deficit[None, :] > 0, v < 1.0 - 1e-9, v > -1.0 + 1e-9)
            cnt = can.sum(0); cnt = np.where(cnt == 0, 1, cnt)
            v = v + can * (deficit / cnt)[None, :]
        out[i*K:(i+1)*K, :D] = v
    return out.astype(np.float32)


def coarsen(actions, k):
    n_blocks = len(actions) // k
    return actions[:n_blocks*k].reshape(n_blocks, k, actions.shape[1]).sum(axis=1)


meta = json.loads(open(JS).read())
with h5py.File(H5) as h:
    eligible = [e for e in meta["episodes"] if len(h[f"traj_{int(e['episode_id'])}"]["actions"]) >= 16][:N]

env = gym.make("PushT-v1", num_envs=1, obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="physx_cuda", reconfiguration_freq=1)
u = env.unwrapped
BASE = {"exact_integral": resample_exact_integral, "cubic_spline": resample_cubic_spline,
        "pchip": resample_pchip, "tac_fold": resample_tac_fold}
ARMS = ["original"] + [a for b in BASE for a in (b, b + "_satfix")]
succ = {a: [] for a in ARMS}; sat = {a: [] for a in ARMS}; ep_ids = []
print(f"[PushT] eligible={len(eligible)} arms={ARMS} k={K}", flush=True)

import torch
with h5py.File(H5) as h:
    for idx, ep in enumerate(eligible):
        ep_id = int(ep["episode_id"])
        a_full = np.asarray(h[f"traj_{ep_id}"]["actions"], np.float32)
        L = (len(a_full) // K) * K
        a = np.clip(a_full[:L], -1, 1).astype(np.float32)
        delta = coarsen(a, K)
        nd = delta.shape[1]
        arms = {"original": a}
        for b, fn in BASE.items():
            p = fn(delta, None, L)
            arms[b] = p
            arms[b + "_satfix"] = satfix(p, delta, K)

        sg = h[f"traj_{ep_id}"]["env_states"]
        state = {g: {n: torch.as_tensor(np.asarray(sg[g][n])[0:1]) for n in sg[g]} for g in sg}

        for arm in ARMS:
            acts = arms[arm]
            assert np.isfinite(acts).all()
            sat[arm].append(float((np.abs(acts[:, :nd]) > 1.0).mean()))
            env.reset(seed=ep["episode_seed"])
            try:
                u.set_state_dict(state)
            except Exception as exc:
                raise RuntimeError(f"set_state_dict failed PushT ep {ep_id}: {exc!r}")
            s = False
            for act in acts:
                _, _, term, trunc, info = env.step(act[None])
                succ_t = info.get("success")
                succ_t = succ_t.cpu().numpy() if hasattr(succ_t, "cpu") else np.asarray(succ_t)
                if bool(succ_t.reshape(-1)[0]):
                    s = True
                term_np = term.cpu().numpy() if hasattr(term, "cpu") else np.asarray(term)
                trunc_np = trunc.cpu().numpy() if hasattr(trunc, "cpu") else np.asarray(trunc)
                if bool(term_np.reshape(-1)[0]) or bool(trunc_np.reshape(-1)[0]):
                    break
            succ[arm].append(s)
        ep_ids.append(ep_id)
        if (idx + 1) % 20 == 0 or idx + 1 == len(eligible):
            print(f"[PushT] [{idx+1}/{len(eligible)}] " + " ".join(f"{a}={sum(succ[a])}" for a in ARMS), flush=True)

env.close()
S = {a: np.asarray(succ[a], bool) for a in ARMS}
print(f"\n=== [PushT] k={K} n={len(ep_ids)} ===")
for a in ARMS:
    print(f"  {a:22s} {S[a].mean()*100:6.2f}% ({S[a].sum()}/{len(S[a])})  saturated_elems={100*np.mean(sat[a]):.2f}%")
print("--- paired contrasts vs ZOH (exact McNemar, 95% CI pp) ---")
res = {"task": "PushT-v1", "k": K, "n": len(ep_ids), "episode_ids": ep_ids,
       "success": {a: [bool(x) for x in succ[a]] for a in ARMS},
       "saturation_frac": {a: sat[a] for a in ARMS}, "contrasts": {}}
zoh = S["exact_integral"]
for a in ARMS:
    if a in ("original", "exact_integral"):
        continue
    a_only, b_only, p = exact_mcnemar(S[a], zoh)
    lo, hi = paired_bootstrap_ci(S[a].astype(float), zoh.astype(float))
    delta_pp = 100 * (S[a].mean() - zoh.mean())
    print(f"  {a:22s} vs ZOH: {delta_pp:+6.2f}pp  McNemar p={p:.4g}  95%CI=[{100*lo:.2f},{100*hi:.2f}]pp")
    res["contrasts"][a] = {"delta_pp": delta_pp, "mcnemar_p": p, "ci95_pp": [100*lo, 100*hi]}
with open(f"{SD}/pusht_satfix_k{K}.json", "w") as fp:
    json.dump(res, fp, indent=2)
print(f"[saved] {SD}/pusht_satfix_k{K}.json")
