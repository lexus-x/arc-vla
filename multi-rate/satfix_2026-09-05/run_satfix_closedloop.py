"""
Pre-declared test of the ACTUATOR-SATURATION mechanism.
Offline finding (before this run): cubic_spline and shape_ridge exceed |a|>1 in 399/399
held-out episodes (~4.2% of elements); the controller clips to [-1,1], silently losing
~1.4-1.6% of commanded displacement -> their 'integral conservation' is not realized.
ZOH never saturates (|S/2|<=1), original never does (pre-clipped).

Fix under test ('satfix'): per block, project (v1,v2) onto {|v1|<=1, |v2|<=1, v1+v2=S}
-- keeps shape where feasible, guarantees the commanded integral is realizable.
Prediction fixed before results: if saturation is THE mechanism, *_satfix arms recover to
>= ZOH. If they still lose to ZOH, saturation is not the whole story. Reported either way.
Also records realized joint trajectories for mechanism diagnostics.
"""
import sys, json, numpy as np, h5py, torch
import gymnasium as gym, mani_skill.envs  # noqa
sys.path.insert(0, "/home/user/Desktop/multi-rate/vla-vault/scratch")
from sweep_maniskill_decimation_ratios import (coarsen_actions, resample_exact_integral,
    resample_cubic_spline, exact_mcnemar, paired_bootstrap_ci)

SD = "/tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad"
W = json.load(open(f"{SD}/shape_ridge_weights.json"))
WEIGHTS = np.array(W["weights_per_dim"]); TEST_IDS = set(W["test_episode_ids_heldout"]); K = 2

def ridge_v1(delta):
    n, D = delta.shape; v1 = np.empty((n, D), np.float32)
    for i in range(n):
        s_im1 = delta[i-1] if i > 0 else np.zeros(D, np.float32)
        s_ip1 = delta[i+1] if i+1 < n else np.zeros(D, np.float32)
        f = np.stack([s_im1, delta[i], s_ip1, np.ones(D, np.float32)], 1)
        v1[i] = np.einsum("df,df->d", f, WEIGHTS)
    return v1

def assemble(v1, delta, gripper, satfix):
    n, D = delta.shape
    if satfix:
        lo = np.maximum(-1.0, delta - 1.0); hi = np.minimum(1.0, delta + 1.0)
        v1 = np.clip(v1, lo, hi)
    v2 = delta - v1
    out = np.empty((2*n, D), np.float32); out[0::2] = v1; out[1::2] = v2
    return np.concatenate([out, np.repeat(gripper, K, 0)], 1).astype(np.float32)

def build_arms(actions, delta, gripper, L):
    spl = resample_cubic_spline(delta, gripper, L)
    spl_v1 = spl[0::2, :7]
    rv1 = ridge_v1(delta)
    return {
        "original": actions,
        "exact_integral": resample_exact_integral(delta, gripper, L),
        "cubic_spline": spl,
        "cubic_spline_satfix": assemble(spl_v1.copy(), delta, gripper, True),
        "shape_ridge": assemble(rv1.copy(), delta, gripper, False),
        "shape_ridge_satfix": assemble(rv1.copy(), delta, gripper, True),
    }

N = int(sys.argv[1]) if len(sys.argv) > 1 else 399
meta = json.loads(open("/home/user/maniskill_data/pick_rl_joint.json").read())
episodes = [e for e in meta["episodes"] if int(e["episode_id"]) in TEST_IDS][:N]
env = gym.make("PickCube-v1", num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")
u = env.unwrapped
ARMS = ["original", "exact_integral", "cubic_spline", "cubic_spline_satfix", "shape_ridge", "shape_ridge_satfix"]
out = {a: {"success": [], "first_grasp": [], "grasp_then_drop": [], "rms_dev_from_original": [], "final_dev_from_original": [], "joint_travel": []} for a in ARMS}
ep_ids = []

with h5py.File("/home/user/maniskill_data/pick_rl_joint.h5") as h:
    for idx, ep in enumerate(episodes):
        ep_id = int(ep["episode_id"])
        a = np.asarray(h[f"traj_{ep_id}"]["actions"], np.float32); L = (len(a)//K)*K
        if L < K: continue
        a = np.clip(a[:L], -1, 1).astype(np.float32)
        delta, gripper = coarsen_actions(a, K, True)
        arms = build_arms(a, delta, gripper, L)
        for k, v in arms.items():
            assert np.isfinite(v).all(), k
            if "satfix" in k: assert np.abs(v[:, :7]).max() <= 1.0 + 1e-6, k
        sg = h[f"traj_{ep_id}"]["env_states"]
        state = {g: {n: torch.as_tensor(np.asarray(sg[g][n])[0:1]) for n in sg[g]} for g in sg}
        q_orig = None
        for arm in ARMS:
            env.reset(seed=ep["episode_seed"]); u.set_state_dict(state)
            succ = False; first_grasp = -1; grasped_ever = False; dropped = False
            qs = []
            for t, act in enumerate(arms[arm]):
                _, _, term, trunc, info = env.step(act[None])
                q = u.agent.robot.get_qpos()[0, :7].cpu().numpy().copy(); qs.append(q)
                g = bool(np.asarray(info.get("is_grasped")).reshape(-1)[0])
                if g and first_grasp < 0: first_grasp = t
                if g: grasped_ever = True
                if grasped_ever and not g and not succ: dropped = True
                if bool(np.asarray(info.get("success")).reshape(-1)[0]): succ = True
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
            qs = np.array(qs)
            if arm == "original": q_orig = qs
            m = min(len(qs), len(q_orig))
            dev = np.linalg.norm(qs[:m] - q_orig[:m], axis=1)
            out[arm]["success"].append(succ); out[arm]["first_grasp"].append(first_grasp)
            out[arm]["grasp_then_drop"].append(dropped)
            out[arm]["rms_dev_from_original"].append(float(np.sqrt((dev**2).mean())))
            out[arm]["final_dev_from_original"].append(float(dev[-1]))
            out[arm]["joint_travel"].append(float(np.abs(np.diff(qs, axis=0)).sum()))
        ep_ids.append(ep_id)
        if (idx+1) % 50 == 0 or idx+1 == len(episodes):
            print(f"[{idx+1}/{len(episodes)}] " + " ".join(f"{a}={sum(out[a]['success'])}" for a in ARMS), flush=True)
env.close()

S = {a: np.asarray(out[a]["success"], bool) for a in ARMS}
print(f"\n=== SUCCESS (PickCube-v1, k=2, held-out n={len(ep_ids)}) ===")
for a in ARMS:
    fg = np.array(out[a]["first_grasp"]); print(f"  {a:20s} {S[a].mean()*100:6.2f}% ({S[a].sum()}/{len(S[a])})  grasped={np.mean(fg>=0)*100:5.1f}%  grasp_then_drop={np.mean(out[a]['grasp_then_drop'])*100:5.1f}%  rms_dev_vs_orig={np.mean(out[a]['rms_dev_from_original']):.4f}  final_dev={np.mean(out[a]['final_dev_from_original']):.4f}  travel={np.mean(out[a]['joint_travel']):.3f}")
print("\n=== PAIRED CONTRASTS (exact McNemar, 95% bootstrap CI pp) ===")
for x, y in [("cubic_spline_satfix","cubic_spline"),("shape_ridge_satfix","shape_ridge"),
             ("cubic_spline_satfix","exact_integral"),("shape_ridge_satfix","exact_integral"),
             ("shape_ridge_satfix","cubic_spline_satfix")]:
    ao, bo, p = exact_mcnemar(S[x], S[y]); lo, hi = paired_bootstrap_ci(S[x], S[y])
    print(f"  {x} vs {y}: {100*(S[x].mean()-S[y].mean()):+.2f}pp  wins {ao}/{bo}  p={p:.4g}  CI=[{100*lo:.2f},{100*hi:.2f}]")
json.dump({"n": len(ep_ids), "episode_ids": ep_ids, "arms": {a: {k: [float(x) if not isinstance(x, (bool, np.bool_)) else bool(x) for x in v] for k, v in out[a].items()} for a in ARMS}},
          open(f"{SD}/pickcube_k2_satfix_results.json", "w"))
print(f"saved -> {SD}/pickcube_k2_satfix_results.json")
