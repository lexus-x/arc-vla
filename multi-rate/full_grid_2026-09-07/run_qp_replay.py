"""Open-loop paired replay: QP resampler vs satfix family. Same code path as
satfix_2026-09-05/run_satfix_general.py (physx_cpu, set_state_dict replay, exact McNemar).
Usage: python run_qp_replay.py PickCube-v1 [N] [K]"""
import sys, json, os, time, numpy as np, h5py, torch
import gymnasium as gym, mani_skill.envs  # noqa
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/home/user/Desktop/multi-rate/vla-vault/scratch")
from sweep_maniskill_decimation_ratios import (TASK_CONFIGS, coarsen_actions, resample_exact_integral,
    resample_cubic_spline, resample_pchip, resample_tac_fold, exact_mcnemar, paired_bootstrap_ci)
from resample_qp import resample_qp, resample_qp_anchor
from resample_bspline2 import resample_bspline_eps
BSPLINE_EPS = 0.005  # pre-registered (PREREG_BSPLINE.md)

task = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
cfg = TASK_CONFIGS[task]

def satfix(pose, delta):
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

def qp_arm(delta, gripper, L, fn=resample_qp):
    pose = fn(delta.astype(np.float64), K)
    if gripper is not None:
        return np.concatenate([pose, np.repeat(gripper, K, axis=0)], axis=1).astype(np.float32)
    return pose.astype(np.float32)

meta = json.loads(open(cfg["json_path"]).read())
with h5py.File(cfg["h5_path"]) as h:
    eligible = [e for e in meta["episodes"] if len(h[f"traj_{int(e['episode_id'])}"]["actions"]) >= 16][:N]
env = gym.make(cfg["env_id"], num_envs=1, obs_mode="state", control_mode=cfg["control_mode"], sim_backend="physx_cpu")
u = env.unwrapped
BASE = {"cubic_spline": resample_cubic_spline, "pchip": resample_pchip, "tac_fold": resample_tac_fold}
ARMS = [a for a in os.environ.get("ARMS", "original,exact_integral,cubic_spline_satfix,pchip_satfix,tac_fold_satfix,qp").split(",")]
succ = {a: [] for a in ARMS}; sat = {a: [] for a in ARMS}; ep_ids = []; nd = None
print(f"[{task}] k={K} eligible={len(eligible)} arms={ARMS}", flush=True); t0 = time.time()
with h5py.File(cfg["h5_path"]) as h:
    for idx, ep in enumerate(eligible):
        ep_id = int(ep["episode_id"])
        a = np.asarray(h[f"traj_{ep_id}"]["actions"], np.float32); L = (len(a)//K)*K
        if L < K: continue
        a = np.clip(a[:L], -1, 1).astype(np.float32)
        delta, gripper = coarsen_actions(a, K, cfg["has_gripper"]); nd = delta.shape[1]
        arms = {"original": a, "exact_integral": resample_exact_integral(delta, gripper, L)}
        for b, fn in BASE.items(): arms[b + "_satfix"] = satfix(fn(delta, gripper, L), delta)
        arms["qp"] = qp_arm(delta, gripper, L)
        if "qp_anchor" in ARMS:
            arms["qp_anchor"] = qp_arm(delta, gripper, L, fn=resample_qp_anchor)
        if "bspline_eps_satfix" in ARMS:
            pose = resample_bspline_eps(delta.astype(np.float64), K, eps=BSPLINE_EPS)
            raw = np.concatenate([pose, np.repeat(gripper, K, axis=0)], 1).astype(np.float32) if gripper is not None else pose.astype(np.float32)
            arms["bspline_eps_satfix"] = satfix(raw, delta)
        # Fair-competition arms: the actual published algorithms, un-repaired by our own satfix
        # mechanism (house rule, PREREG_QP_ANCHOR_2ND_TASK.md:42 -- lending satfix to competitors
        # overstates them). Only clipped at step time (below), like every other arm.
        if "spline" in ARMS:
            arms["spline"] = BASE["cubic_spline"](delta, gripper, L)
        if "bspline_eps_raw" in ARMS:
            pose = resample_bspline_eps(delta.astype(np.float64), K, eps=BSPLINE_EPS)
            arms["bspline_eps_raw"] = np.concatenate([pose, np.repeat(gripper, K, axis=0)], 1).astype(np.float32) if gripper is not None else pose.astype(np.float32)
        sg = h[f"traj_{ep_id}"]["env_states"]
        state = {g: {n: torch.as_tensor(np.asarray(sg[g][n])[0:1]) for n in sg[g]} for g in sg}
        for arm in ARMS:
            acts = arms[arm]; assert np.isfinite(acts).all() and acts.shape == a.shape, (arm, acts.shape, a.shape)
            sat[arm].append(float((np.abs(acts[:, :nd]) > 1.0 + 1e-6).mean()))
            env.reset(seed=ep["episode_seed"]); u.set_state_dict(state)
            s = False
            for act in acts:
                # clip at step time only (harness.py:135 semantics) -- no block-sum redistribution,
                # so raw spline/bspline_eps_raw are genuinely un-repaired, not silently satfixed.
                _, _, term, trunc, info = env.step(np.clip(act, -1, 1)[None])
                if bool(np.asarray(info.get("success")).reshape(-1)[0]): s = True
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
            succ[arm].append(s)
        ep_ids.append(ep_id)
        if (idx+1) % 100 == 0 or idx+1 == len(eligible):
            print(f"[{task}] k={K} [{idx+1}/{len(eligible)}] " + " ".join(f"{a}={sum(succ[a])}" for a in ARMS) + f" ({time.time()-t0:.0f}s)", flush=True)
env.close()
S = {a: np.asarray(succ[a], bool) for a in ARMS}
print(f"\n=== [{task}] k={K} n={len(ep_ids)} ===")
for a in ARMS: print(f"  {a:22s} {S[a].mean()*100:6.2f}% ({S[a].sum()}/{len(S[a])})  saturated_elems={100*np.mean(sat[a]):.2f}%")
res = {"task": task, "k": K, "n": len(ep_ids), "episode_ids": ep_ids, "arms": ARMS,
       "success": {a: [bool(x) for x in succ[a]] for a in ARMS}, "saturation_frac": {a: sat[a] for a in ARMS}, "contrasts": {}}
for ref in [r for r in ("exact_integral", "cubic_spline_satfix", "tac_fold_satfix", "pchip_satfix", "bspline_eps_satfix", "spline", "bspline_eps_raw") if r in ARMS]:
    for a in ARMS:
        if a in ("original", ref): continue
        ao, bo, p = exact_mcnemar(S[a], S[ref]); lo, hi = paired_bootstrap_ci(S[a], S[ref])
        d = 100*(S[a].mean()-S[ref].mean())
        res["contrasts"][f"{a}_vs_{ref}"] = {"delta_pp": d, "wins": ao, "losses": bo, "p": p, "ci": [100*lo, 100*hi]}
        if a in ("qp", "qp_anchor"): print(f"  {a} vs {ref:20s}: {d:+.2f}pp  wins {ao}/{bo}  p={p:.3g}  CI=[{100*lo:.2f},{100*hi:.2f}]")
out = f"{HERE}/{os.environ.get('OUT_PREFIX', 'qp_replay')}_{task}_k{K}.json"; json.dump(res, open(out, "w")); print(f"saved -> {out}")
