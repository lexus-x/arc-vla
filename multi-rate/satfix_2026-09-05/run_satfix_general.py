"""
Generalization + confirmation of the saturation-projection ('satfix') finding.
Arms: original, exact_integral(ZOH), cubic_spline, pchip, tac_fold, and *_satfix for each.
satfix = per block project (v1,v2) onto {|v1|<=1,|v2|<=1,v1+v2=S}. No learned parameters,
so ALL eligible episodes are fair game (no train/test split needed).
Pre-declared: every task/cell is reported, wins and losses. k=2 only (vault's discriminating k;
k>=4 floor-collapses 2/4 tasks per gate_prereg_SEALED.json).
Usage: python run_satfix_general.py <TASK> [n_episodes]
"""
import sys, json, numpy as np, h5py, torch, os
import gymnasium as gym, mani_skill.envs  # noqa
sys.path.insert(0, "/home/user/Desktop/multi-rate/vla-vault/scratch")
from sweep_maniskill_decimation_ratios import (TASK_CONFIGS, coarsen_actions, resample_exact_integral,
    resample_cubic_spline, resample_pchip, resample_tac_fold, exact_mcnemar, paired_bootstrap_ci)
SD = "/tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad"
task = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
cfg = TASK_CONFIGS[task]
if not (os.path.exists(cfg["h5_path"]) and os.path.exists(cfg["json_path"])):
    print(f"[{task}] MISSING DATA: {cfg['h5_path']}"); sys.exit(0)
nd = None
def satfix(pose, delta):
    """Project each block's k-vector onto {|v_i|<=1, sum v_i = S}: clip, redistribute the
    deficit equally over elements that can still move in the needed direction, repeat.
    Feasible whenever |S|<=k, which holds because S is a sum of k clipped values. For k=2
    this reduces exactly to v1=clip(v1, max(-1,S-1), min(1,S+1)), v2=S-v1."""
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
meta = json.loads(open(cfg["json_path"]).read())
with h5py.File(cfg["h5_path"]) as h:
    eligible = [e for e in meta["episodes"] if len(h[f"traj_{int(e['episode_id'])}"]["actions"]) >= 16][:N]
env = gym.make(cfg["env_id"], num_envs=1, obs_mode="state", control_mode=cfg["control_mode"], sim_backend="physx_cpu")
u = env.unwrapped
BASE = {"exact_integral": resample_exact_integral, "cubic_spline": resample_cubic_spline, "pchip": resample_pchip, "tac_fold": resample_tac_fold}
ARMS = ["original"] + [a for b in BASE for a in (b, b + "_satfix")]
succ = {a: [] for a in ARMS}; sat = {a: [] for a in ARMS}; ep_ids = []
print(f"[{task}] eligible={len(eligible)} arms={ARMS}", flush=True)
with h5py.File(cfg["h5_path"]) as h:
    for idx, ep in enumerate(eligible):
        ep_id = int(ep["episode_id"])
        a = np.asarray(h[f"traj_{ep_id}"]["actions"], np.float32); L = (len(a)//K)*K
        if L < K: continue
        a = np.clip(a[:L], -1, 1).astype(np.float32)
        delta, gripper = coarsen_actions(a, K, cfg["has_gripper"])
        nd = delta.shape[1]
        arms = {"original": a}
        for b, fn in BASE.items():
            p = fn(delta, gripper, L); arms[b] = p; arms[b + "_satfix"] = satfix(p, delta)
        sg = h[f"traj_{ep_id}"]["env_states"]
        state = {g: {n: torch.as_tensor(np.asarray(sg[g][n])[0:1]) for n in sg[g]} for g in sg}
        for arm in ARMS:
            acts = arms[arm]; assert np.isfinite(acts).all()
            sat[arm].append(float((np.abs(acts[:, :nd]) > 1.0).mean()))
            env.reset(seed=ep["episode_seed"])
            try: u.set_state_dict(state)
            except Exception as exc: raise RuntimeError(f"set_state_dict failed {task} ep {ep_id}: {exc!r}")
            s = False
            for act in acts:
                _, _, term, trunc, info = env.step(act[None])
                if bool(np.asarray(info.get("success")).reshape(-1)[0]): s = True
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
            succ[arm].append(s)
        ep_ids.append(ep_id)
        if (idx+1) % 100 == 0 or idx+1 == len(eligible):
            print(f"[{task}] [{idx+1}/{len(eligible)}] " + " ".join(f"{a}={sum(succ[a])}" for a in ARMS), flush=True)
env.close()
S = {a: np.asarray(succ[a], bool) for a in ARMS}
print(f"\n=== [{task}] k={K} n={len(ep_ids)} ===")
for a in ARMS: print(f"  {a:22s} {S[a].mean()*100:6.2f}% ({S[a].sum()}/{len(S[a])})  saturated_elems={100*np.mean(sat[a]):.2f}%")
print(f"--- paired contrasts vs ZOH (exact McNemar, 95% CI pp) ---")
res = {"task": task, "k": K, "n": len(ep_ids), "episode_ids": ep_ids, "success": {a: [bool(x) for x in succ[a]] for a in ARMS}, "saturation_frac": {a: sat[a] for a in ARMS}, "contrasts": {}}
for a in ARMS:
    if a in ("original", "exact_integral"): continue
    ao, bo, p = exact_mcnemar(S[a], S["exact_integral"]); lo, hi = paired_bootstrap_ci(S[a], S["exact_integral"])
    d = 100*(S[a].mean()-S["exact_integral"].mean())
    res["contrasts"][f"{a}_vs_zoh"] = {"delta_pp": d, "wins": ao, "losses": bo, "p": p, "ci": [100*lo, 100*hi]}
    print(f"  {a:22s} vs ZOH: {d:+.2f}pp  wins {ao}/{bo}  p={p:.3g}  CI=[{100*lo:.2f},{100*hi:.2f}]")
json.dump(res, open(f"{SD}/satfix_{task}_k{K}.json", "w"))
print(f"saved -> {SD}/satfix_{task}_k{K}.json")
