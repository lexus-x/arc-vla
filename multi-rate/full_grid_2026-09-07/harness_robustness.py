#!/usr/bin/env python3
"""Empirical State-Space Robustness Evaluation Harness for RoboMimic.

Evaluates Diffusion Policy under paired observation / sensor noise across:
  - Tasks: lift, can, square
  - Speedups: k in {1, 2, 4, 8}
  - Arms: native, zoh, spline_satfix, bspline_satfix, qp
  - Sensor Noise: --obs_noise (e.g. 0.05 = 5% normalized state jitter)

Zero estimations: executes real closed-loop rollouts (n=100) and saves exact
success booleans and McNemar statistics to result_dp_{task}_k{k}_noise{std}.json.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dp_min import DiffusionPolicy, MinMax, make_chunks, train
import resample_math
from resample_math import coarsen_gripper_transitions, decimate_and_resample, gripper_sync
from resample_qp import resample_qp, resample_qp_anchor
from resample_bspline2 import resample_bspline_eps

# Register all resamplers
resample_math.RESAMPLERS["qp"] = resample_qp
resample_math.RESAMPLERS["qp_anchor"] = resample_qp_anchor
resample_math.RESAMPLERS["bspline_eps_satfix"] = resample_math._satfix_wrap(
    lambda b, k: resample_bspline_eps(b, k, eps=0.005)
)
resample_math.RESAMPLERS["bspline_satfix"] = resample_math.RESAMPLERS["bspline_eps_satfix"]

ROBOMIMIC = {"lift": 200, "can": 400, "square": 400}
RM_OBS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
DEFAULT_ARMS = ["native", "zoh", "spline_satfix", "bspline_satfix", "qp"]


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def exact_mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    ao, bo = int((a & ~b).sum()), int((~a & b).sum())
    d = ao + bo
    if d == 0:
        return ao, bo, 1.0
    return ao, bo, float(min(1.0, 2 * sum(math.comb(d, i) for i in range(min(ao, bo) + 1)) / (2 ** d)))


def apply_arm(chunk, arm, n_hold, k):
    if arm == "native":
        return chunk
    nd = chunk.shape[1] - n_hold
    out = decimate_and_resample(chunk[:, :nd], k, arm)
    if n_hold:
        T = len(chunk)
        nb = T // k
        if nb > 0:
            g = np.repeat(chunk[:nb * k:k, nd:], k, 0)
            if nb * k < T:
                g = np.concatenate([g, chunk[nb * k:, nd:]], 0)
        else:
            g = chunk[:, nd:]
        out = np.concatenate([out, g], 1)
        if arm == "gripper_sync":
            transition_summary = coarsen_gripper_transitions(chunk[:, nd], k)
            out = gripper_sync(out, transition_summary, n_hold, k)
    return out.astype(np.float32)


class RoboMimicSim:
    def __init__(self, task):
        import robosuite, h5py
        self.f = h5py.File(f"/home/user/robomimic_data/{task}.hdf5", "r")
        env_meta = json.loads(self.f["data"].attrs["env_args"])
        self.env = robosuite.make(env_meta["env_name"], **env_meta["env_kwargs"])
        self.has_gripper = True
        self.n_hold = 1
        self.max_steps = ROBOMIMIC[task]
        self.eligible = [d for d in self.f["data"].keys() if self.f["data"][d]["actions"].shape[0] >= 16]

    _LIVE = {"object": "object-state"}

    def _obs_from_dict(self, od):
        return np.concatenate([np.asarray(od[self._LIVE.get(k, k)], np.float32).reshape(-1) for k in RM_OBS])

    def split(self, n_train, n_eval):
        return self.eligible[:n_train], list(range(n_eval))

    def reset_to(self, seed):
        np.random.seed(10_000 + seed)
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

    def close(self):
        self.env.close()
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(ROBOMIMIC))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--obs_noise", type=float, default=0.05, help="Gaussian noise std added to normalized obs")
    ap.add_argument("--arms", default=None, help="Comma-separated arms; default: native,zoh,spline_satfix,bspline_satfix,qp")
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.n_train, args.n_eval = 50, 3, 2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arms = [a.strip() for a in args.arms.split(",")] if args.arms else DEFAULT_ARMS
    k = args.k

    sim = RoboMimicSim(args.task)
    train_eps, eval_eps = sim.split(args.n_train, args.n_eval)

    cache = f"{HERE}/demos_{args.task}_{args.n_train}.npz"
    if os.path.exists(cache) and not args.smoke:
        z = np.load(cache, allow_pickle=True)
        O, A = list(z["O"]), list(z["A"])
        print(f"[data] loaded {cache}")
    else:
        O, A = sim.harvest(train_eps)
        if not args.smoke:
            tmp = f"{cache[:-4]}.{os.getpid()}.tmp.npz"
            np.savez(tmp, O=np.array(O, dtype=object), A=np.array(A, dtype=object))
            os.replace(tmp, cache)

    onorm, anorm = MinMax(np.concatenate(O)), MinMax(np.concatenate(A))
    chunks = [make_chunks(onorm.norm(o), anorm.norm(a), 2, 16) for o, a in zip(O, A)]
    OC, AC = np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])

    ckpt = f"{HERE}/dp_{args.task}" + ("_smoke" if args.smoke else "") + ".pt"
    policy = DiffusionPolicy(OC.shape[-1], AC.shape[-1], horizon=16)
    if os.path.exists(ckpt) and not args.smoke:
        policy.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
        policy.to(dev).eval()
        print(f"[train] loaded {ckpt}")
    else:
        policy = train(policy, OC, AC, steps=args.steps, dev=dev)
        torch.save(policy.state_dict(), ckpt)
        print(f"[train] saved {ckpt}")
    policy.eval()

    succ = {a: [] for a in arms}
    sat = {a: [] for a in arms}
    t_eval = time.time()

    noise_tag = f"noise{int(round(args.obs_noise * 100)):02d}"
    print(f"\n[eval] starting {args.task} (k={k}, n={len(eval_eps)}, obs_noise={args.obs_noise}, arms={arms})...", flush=True)

    for ei, ep in enumerate(eval_eps):
        for arm in arms:
            obs = sim.reset_to(ep)
            hist = [obs, obs]
            s = False
            steps = 0
            sat_e = []
            replan = 0

            while steps < sim.max_steps and not s:
                o_norm = onorm.norm(np.stack(hist[-2:]))
                if args.obs_noise > 0:
                    rng = np.random.default_rng(2_000_003 * ei + replan)
                    noise = rng.normal(0.0, args.obs_noise, size=o_norm.shape).astype(np.float32)
                    o_input = np.clip(o_norm + noise, -1.0, 1.0)
                else:
                    o_input = o_norm

                o = torch.as_tensor(o_input[None], device=dev)
                torch.manual_seed(1_000_003 * ei + replan)
                pred = anorm.denorm(policy.sample(o).cpu().numpy()[0])
                chunk = pred[:8]
                sat_e.append(float((np.abs(chunk) > 1).mean()))
                exec_chunk = apply_arm(np.clip(chunk, -1, 1).astype(np.float32), arm, sim.n_hold, k)

                for a in exec_chunk:
                    obs, s_t, done = sim.step(a)
                    hist.append(obs)
                    steps += 1
                    s |= s_t
                    if s or done or steps >= sim.max_steps:
                        break
                replan += 1

            succ[arm].append(s)
            sat[arm].append(float(np.mean(sat_e)))

        if (ei + 1) % 10 == 0 or ei + 1 == len(eval_eps):
            rates = " ".join(f"{a}={sum(succ[a])}/{len(succ[a])}" for a in arms)
            print(f"[eval] [{ei+1}/{len(eval_eps)}] {rates}", flush=True)

    sim.close()
    print(f"[eval] completed in {time.time()-t_eval:.1f}s", flush=True)

    S = {a: np.asarray(succ[a], bool) for a in arms}
    res = {
        "task": args.task,
        "policy": "dp",
        "k": k,
        "obs_noise": args.obs_noise,
        "n": len(eval_eps),
        "arms": arms,
        "success": {a: succ[a] for a in arms},
        "policy_raw_sat_frac": {a: float(np.mean(sat[a])) for a in arms},
        "contrasts": {},
    }

    ref = "zoh" if "zoh" in S and k > 1 else ("native" if "native" in S else arms[0])
    print(f"\n=== {args.task.upper()} ROBUSTNESS (k={k}, noise={args.obs_noise}, n={len(eval_eps)}) ===")
    for a in arms:
        rate = 100 * S[a].mean()
        line = f"  {a:18s}: {rate:6.1f}% ({S[a].sum()}/{len(S[a])})"
        res["contrasts"][a] = {}
        for other in ["native", "zoh", "bspline_satfix"]:
            if other in S and other != a:
                ao, bo, p = exact_mcnemar(S[a], S[other])
                d = 100 * (S[a].mean() - S[other].mean())
                res["contrasts"][a][f"vs_{other}"] = {"delta_pp": d, "p": p, "a_only": ao, "other_only": bo}
                line += f" | vs {other}: {d:+5.1f}pp (p={p:.3g})"
        print(line)

    tag = f"dp_{args.task}_k{k}_{noise_tag}{args.suffix}" + ("_smoke" if args.smoke else "")
    out = f"{HERE}/result_{tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
