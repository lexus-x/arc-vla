"""Closed-loop Diffusion Policy + resampler eval. One script, two sim backends:
  ManiSkill  (env `ms3`):               python harness.py PickCube-v1 | PushT-v1
  RoboMimic  (env `vla_smolvla_libero`): python harness.py lift | can | square

Protocol (pre-registered, do not tune after seeing results):
  * DP = dp_min.DiffusionPolicy, state-only, n_obs=2, horizon=16, execute 8, EMA weights, DDIM-10.
  * Train on the first N_TRAIN demos; eval from the initial states of N_EVAL *held-out* demos.
  * Arms, all applied to the SAME executed 8-step chunk after clipping to [-1,1]:
      native (k=1, chunk as-is) + {zoh, spline, spline_satfix, pchip, pchip_satfix,
      tac_fold, tac_fold_satfix} at k=2 on the delta dims; gripper dim (if any) causal-held.
  * Paired: same init state per episode across arms, and the DP's sampling noise is re-seeded
    per (episode, replan) so arms see identical policy proposals whenever their obs agree.
  * Stats: exact McNemar vs zoh AND vs native, per arm. Loss column always reported.
Usage: python harness.py TASK [--steps 30000] [--n_train 200] [--n_eval 100] [--smoke]
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from dp_min import DiffusionPolicy, MinMax, make_chunks, train
from resample_math import decimate_and_resample

K = 2
ARMS = ["native", "zoh", "spline", "spline_satfix", "pchip", "pchip_satfix", "tac_fold", "tac_fold_satfix"]
MANISKILL = {"PickCube-v1": dict(h5="pick_rl_joint", ctrl="pd_joint_delta_pos", gripper=True, max_steps=100),
             "PushT-v1":    dict(h5="pusht_rl",       ctrl="pd_ee_delta_pose",    gripper=False, max_steps=150)}
ROBOMIMIC = {"lift": 200, "can": 400, "square": 400}  # max steps (DP paper)
RM_OBS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]  # DP low-dim obs


def to_np(x): return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def exact_mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    ao, bo = int((a & ~b).sum()), int((~a & b).sum()); d = ao + bo
    if d == 0: return ao, bo, 1.0
    return ao, bo, float(min(1.0, 2 * sum(math.comb(d, i) for i in range(min(ao, bo) + 1)) / 2 ** d))


def apply_arm(chunk, arm, has_gripper):
    """chunk (T,A) clipped. Resample delta dims at k=2; causal-hold gripper."""
    if arm == "native": return chunk
    nd = chunk.shape[1] - (1 if has_gripper else 0)
    out = decimate_and_resample(chunk[:, :nd], K, arm)
    if has_gripper:
        T = len(chunk); nb = T // K
        g = np.repeat(chunk[:nb * K:K, nd:], K, 0)
        if nb * K < T: g = np.concatenate([g, chunk[nb * K:, nd:]], 0)
        out = np.concatenate([out, g], 1)
    return out.astype(np.float32)


# ------------------------------------------------------------------ sim adapters
class ManiSkillSim:
    def __init__(self, task):
        import gymnasium as gym, mani_skill.envs  # noqa
        cfg = MANISKILL[task]; self.cfg = cfg
        self.env = gym.make(task, num_envs=1, obs_mode="state", control_mode=cfg["ctrl"],
                            sim_backend="physx_cuda", reconfiguration_freq=1)
        self.u = self.env.unwrapped; self.has_gripper = cfg["gripper"]; self.max_steps = cfg["max_steps"]
        self.meta = json.load(open(f"/home/user/maniskill_data/{cfg['h5']}.json"))["episodes"]
        self.h5 = h5py.File(f"/home/user/maniskill_data/{cfg['h5']}.h5", "r")
        self.eligible = [e for e in self.meta if len(self.h5[f"traj_{int(e['episode_id'])}"]["actions"]) >= 16]

    def _state(self, ep):
        sg = self.h5[f"traj_{int(ep['episode_id'])}"]["env_states"]
        return {g: {n: torch.as_tensor(np.asarray(sg[g][n])[0:1]) for n in sg[g]} for g in sg}

    def split(self, n_train, n_eval):  # plenty of demos -> eval from held-out demo init states
        eps = self.eligible[:n_train], self.eligible[n_train:n_train + n_eval]
        assert len(eps[1]) == n_eval, f"not enough held-out demos: {len(eps[1])}"
        return eps

    def reset_to(self, ep):
        self.env.reset(seed=ep["episode_seed"]); self.u.set_state_dict(self._state(ep))
        return to_np(self.u.get_obs()).reshape(-1).astype(np.float32)

    def step(self, a):
        obs, _, term, trunc, info = self.env.step(a[None])
        s = bool(to_np(info.get("success")).reshape(-1)[0])
        done = bool(to_np(term).reshape(-1)[0]) or bool(to_np(trunc).reshape(-1)[0])
        return to_np(obs).reshape(-1).astype(np.float32), s, done

    def harvest(self, eps):
        """No obs stored in these h5s -> re-simulate each demo with its recorded (pre-clip) actions."""
        O, A = [], []
        for i, ep in enumerate(eps):
            acts = np.asarray(self.h5[f"traj_{int(ep['episode_id'])}"]["actions"], np.float32)
            obs = self.reset_to(ep); ob = []
            for t in range(len(acts)):
                ob.append(obs); obs, _, done = self.step(np.clip(acts[t], -1, 1))
                if done: break
            n = len(ob); O.append(np.stack(ob)); A.append(acts[:n])  # train on the RL policy's raw output
            if (i + 1) % 50 == 0: print(f"[harvest] {i+1}/{len(eps)}", flush=True)
        return O, A

    def close(self): self.env.close(); self.h5.close()


class RoboMimicSim:
    def __init__(self, task):
        import robosuite
        self.f = h5py.File(f"/home/user/robomimic_data/{task}.hdf5", "r")
        env_meta = json.loads(self.f["data"].attrs["env_args"])
        self.env = robosuite.make(env_meta["env_name"], **env_meta["env_kwargs"])
        self.has_gripper = True; self.max_steps = ROBOMIMIC[task]
        self.eligible = [d for d in self.f["data"].keys() if self.f["data"][d]["actions"].shape[0] >= 16]

    _LIVE = {"object": "object-state"}  # robomimic hdf5 key -> live robosuite obs key
    def _obs_from_dict(self, od): return np.concatenate([np.asarray(od[self._LIVE.get(k, k)], np.float32).reshape(-1) for k in RM_OBS])

    def split(self, n_train, n_eval):  # eval "episodes" are reset seeds, not demos
        return self.eligible[:n_train], list(range(n_eval))

    def reset_to(self, seed):
        np.random.seed(10_000 + seed)  # robosuite placement samplers use the global np RNG
        od = self.env.reset()
        return self._obs_from_dict(od)

    def step(self, a):
        od, _, _, _ = self.env.step(a)
        return self._obs_from_dict(od), bool(self.env._check_success()), False

    def harvest(self, eps):
        O, A = [], []
        for d in eps:
            g = self.f["data"][d]
            O.append(np.concatenate([np.asarray(g["obs"][k], np.float32).reshape(len(g["actions"]), -1) for k in RM_OBS], 1))
            A.append(np.asarray(g["actions"], np.float32))
        return O, A

    def close(self): self.env.close(); self.f.close()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--n_train", type=int, default=200); ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    if args.smoke: args.steps, args.n_train, args.n_eval = 50, 3, 2
    dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    sim = ManiSkillSim(args.task) if args.task in MANISKILL else RoboMimicSim(args.task)
    tag = args.task + ("_smoke" if args.smoke else "")
    train_eps, eval_eps = sim.split(args.n_train, args.n_eval)

    t0 = time.time(); O, A = sim.harvest(train_eps); print(f"[data] {len(O)} demos harvested in {time.time()-t0:.0f}s", flush=True)
    onorm, anorm = MinMax(np.concatenate(O)), MinMax(np.concatenate(A))
    chunks = [make_chunks(onorm.norm(o), anorm.norm(a), 2, 16) for o, a in zip(O, A)]
    OC, AC = np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])
    print(f"[data] {len(OC)} windows, obs_dim={OC.shape[-1]}, act_dim={AC.shape[-1]}, "
          f"raw |a|>1 frac={float((np.abs(np.concatenate(A))>1).mean()):.3f}", flush=True)

    ckpt = f"{HERE}/dp_{tag}.pt"
    policy = DiffusionPolicy(OC.shape[-1], AC.shape[-1])
    if os.path.exists(ckpt) and not args.smoke:
        policy.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True)); policy.to(dev).eval(); print(f"[train] loaded {ckpt}")
    else:
        policy = train(policy, OC, AC, steps=args.steps, dev=dev)
        torch.save(policy.state_dict(), ckpt); print(f"[train] saved {ckpt}")
    policy.eval()

    # ---- eval
    succ = {a: [] for a in ARMS}; sat = {a: [] for a in ARMS}; t_eval = time.time()
    for ei, ep in enumerate(eval_eps):
        for arm in ARMS:
            obs = sim.reset_to(ep); hist = [obs, obs]; s = False; steps = 0; sat_e = []
            replan = 0
            while steps < sim.max_steps and not s:
                o = torch.as_tensor(onorm.norm(np.stack(hist[-2:]))[None], device=dev)
                torch.manual_seed(1_000_003 * ei + replan)  # paired noise across arms
                chunk = anorm.denorm(policy.sample(o).cpu().numpy()[0])[:8]
                sat_e.append(float((np.abs(chunk) > 1).mean()))
                chunk = np.clip(chunk, -1, 1).astype(np.float32)
                for a in apply_arm(chunk, arm, sim.has_gripper):
                    obs, s_t, done = sim.step(a); hist.append(obs); steps += 1; s |= s_t
                    if s or done or steps >= sim.max_steps: break
                replan += 1
            succ[arm].append(s); sat[arm].append(float(np.mean(sat_e)))
        if (ei + 1) % 10 == 0 or ei + 1 == len(eval_eps):
            print(f"[eval] [{ei+1}/{len(eval_eps)}] " + " ".join(f"{a}={sum(succ[a])}" for a in ARMS), flush=True)
    sim.close(); print(f"[eval] wall {time.time()-t_eval:.0f}s for {len(eval_eps)} eps x {len(ARMS)} arms", flush=True)

    S = {a: np.asarray(succ[a], bool) for a in ARMS}
    print(f"\n=== {args.task} closed-loop DP, k={K}, n={len(eval_eps)} ===")
    res = {"task": args.task, "k": K, "n": len(eval_eps), "train_steps": args.steps, "arms": ARMS,
           "success": {a: succ[a] for a in ARMS}, "policy_raw_sat_frac": {a: float(np.mean(sat[a])) for a in ARMS}, "contrasts": {}}
    for a in ARMS:
        line = f"  {a:16s} {100*S[a].mean():6.1f}% ({S[a].sum()}/{len(S[a])})"
        res["contrasts"][a] = {}
        for ref in ("zoh", "native"):
            if a == ref: continue
            ao, bo, p = exact_mcnemar(S[a], S[ref]); d = 100 * (S[a].mean() - S[ref].mean())
            res["contrasts"][a][f"vs_{ref}"] = {"delta_pp": d, "p": p, "a_only": ao, "ref_only": bo}
            line += f"  | vs {ref}: {d:+6.1f}pp p={p:.3g}"
        print(line)
    print(f"  policy raw |a|>1 frac (pre-clip): {res['policy_raw_sat_frac']['native']:.3f}")
    out = f"{HERE}/result_{tag}.json"; json.dump(res, open(out, "w"), indent=2); print(f"[saved] {out}")


if __name__ == "__main__":
    main()
