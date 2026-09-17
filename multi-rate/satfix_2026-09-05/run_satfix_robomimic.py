"""
satfix on RoboMimic (real robosuite envs, real human/machine demos), open-loop demo replay --
same methodology as run_satfix_general.py (ManiSkill): decimate the recorded expert actions by
k, reconstruct via {exact_integral(ZOH), cubic_spline, pchip, tac_fold} +/- satfix, replay into
the actual robosuite env from the demo's initial physical state, check success.

robomimic 0.2.0's own EnvRobosuite wrapper hard-imports the long-dead mujoco_py, so this drives
robosuite directly using the env_args already stored in each hdf5's data.attrs (verified
working: Lift demo_0 replays byte-identical, success=True). All 4 tasks use robosuite's
OSC_POSE controller with input_max=1/input_min=-1/control_delta=True -- the same saturating
per-step delta-clip pattern as ManiSkill's pd_joint_delta_pos, so satfix's premise applies
unmodified.

Math (coarsen_actions/resample_*/exact_mcnemar/paired_bootstrap_ci) ported verbatim from
/home/user/Desktop/multi-rate/vla-vault/scratch/sweep_maniskill_decimation_ratios.py --
inlined rather than imported because that module also imports mani_skill at load time
(needed there for env registration, irrelevant and unavailable here).

Usage: python run_satfix_robomimic.py <lift|can|square|tool_hang> [n_episodes] [k]
"""
import sys, json, math, os
import numpy as np
import h5py
import robosuite
from scipy.interpolate import CubicSpline, PchipInterpolator

SD = "/tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad"
DATA_DIR = "/home/user/robomimic_data"

task = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2

hdf5_path = f"{DATA_DIR}/{task}.hdf5"
if not os.path.exists(hdf5_path):
    print(f"[{task}] MISSING DATA: {hdf5_path}"); sys.exit(0)


def compute_akima_slopes(x, y):
    n, d = y.shape
    if n <= 2:
        dx = x[1] - x[0]
        slope = (y[1:] - y[:-1]) / dx
        return np.repeat(slope, n, axis=0)
    dx = np.diff(x)
    m = np.diff(y, axis=0) / dx[:, None]
    m_pad = np.empty((n + 3, d), dtype=y.dtype)
    m_pad[2:-2] = m
    m_pad[1] = 2.0 * m[0] - m[1]
    m_pad[0] = 2.0 * m_pad[1] - m[0]
    m_pad[-2] = 2.0 * m[-1] - m[-2]
    m_pad[-1] = 2.0 * m_pad[-2] - m[-1]
    dm = np.abs(np.diff(m_pad, axis=0))
    w1 = dm[2:]; w2 = dm[:-2]
    weights_sum = w1 + w2
    zero_mask = weights_sum < 1e-12
    weights_sum_safe = np.where(zero_mask, 1.0, weights_sum)
    slopes = (w1 * m_pad[1:-2] + w2 * m_pad[2:-1]) / weights_sum_safe
    slopes = np.where(zero_mask, 0.5 * (m_pad[1:-2] + m_pad[2:-1]), slopes)
    return slopes


def eval_hermite_cubic(x0, x1, y0, y1, d0, d1, x):
    h = x1 - x0
    t = (x - x0) / h
    t2 = t * t; t3 = t2 * t
    h00 = (2.0 * t3 - 3.0 * t2 + 1.0)[:, None]
    h10 = (t3 - 2.0 * t2 + t)[:, None] * h
    h01 = (-2.0 * t3 + 3.0 * t2)[:, None]
    h11 = (t3 - t2)[:, None] * h
    return h00 * y0[None, :] + h10 * d0[None, :] + h01 * y1[None, :] + h11 * d1[None, :]


def coarsen_actions(actions, k, has_gripper=True):
    n_blocks = len(actions) // k
    truncated = actions[: n_blocks * k]
    reshaped = truncated.reshape(n_blocks, k, actions.shape[1])
    if has_gripper:
        delta = reshaped[:, :, :-1].sum(axis=1)
        gripper = reshaped[:, 0, -1:]
        return delta, gripper
    return reshaped.sum(axis=1), None


def resample_exact_integral(delta, gripper, target_len):
    ratio = target_len // len(delta)
    pose = np.repeat(delta / ratio, ratio, axis=0)
    if gripper is not None:
        grip = np.repeat(gripper, ratio, axis=0)
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose.astype(np.float32)


def resample_cubic_spline(delta, gripper, target_len):
    source_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0)
    cs = CubicSpline(source_t, cumulative, axis=0)
    target_t = np.linspace(0.0, float(source_len), target_len + 1)
    pose = np.diff(cs(target_t), axis=0).astype(np.float32)
    if gripper is not None:
        idx = np.minimum(np.floor(np.arange(target_len) * source_len / target_len).astype(int), source_len - 1)
        return np.concatenate([pose, gripper[idx]], axis=1).astype(np.float32)
    return pose


def resample_pchip(delta, gripper, target_len):
    source_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0)
    pchip = PchipInterpolator(source_t, cumulative, axis=0)
    target_t = np.linspace(0.0, float(source_len), target_len + 1)
    pose = np.diff(pchip(target_t), axis=0).astype(np.float32)
    if gripper is not None:
        idx = np.minimum(np.floor(np.arange(target_len) * source_len / target_len).astype(int), source_len - 1)
        return np.concatenate([pose, gripper[idx]], axis=1).astype(np.float32)
    return pose


def resample_tac_fold(delta, gripper, target_len):
    source_len = len(delta)
    ratio = target_len // source_len
    d = delta.shape[1]
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, d), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0)
    slopes = compute_akima_slopes(source_t, cumulative)
    resampled_cum = np.empty((target_len + 1, d), dtype=np.float64)
    resampled_cum[0] = cumulative[0]
    for i in range(source_len):
        x0, x1 = source_t[i], source_t[i + 1]
        y0, y1 = cumulative[i], cumulative[i + 1]
        d0, d1 = slopes[i], slopes[i + 1]
        sub_t = np.linspace(x0, x1, ratio + 1)[1:]
        resampled_cum[i * ratio + 1:(i + 1) * ratio + 1] = eval_hermite_cubic(x0, x1, y0, y1, d0, d1, sub_t)
    pose = np.diff(resampled_cum, axis=0).astype(np.float32)
    if gripper is not None:
        grip = np.repeat(gripper, ratio, axis=0)
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose


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


def exact_mcnemar(a, b):
    a_only = int((a & ~b).sum()); b_only = int((~a & b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    return a_only, b_only, float(min(1.0, 2.0 * tail / (2 ** discordant)))


def paired_bootstrap_ci(a, b, n_boot=10_000, seed=20260819):
    rng = np.random.default_rng(seed)
    diff = a.astype(np.float64) - b.astype(np.float64)
    draws = rng.choice(diff, size=(n_boot, len(diff)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


f = h5py.File(hdf5_path, "r")
env_meta = json.loads(f["data"].attrs["env_args"])
env = robosuite.make(env_meta["env_name"], **env_meta["env_kwargs"])

demo_names = list(f["data"].keys())
eligible = [d for d in demo_names if (f["data"][d]["actions"].shape[0] // K) * K >= 16][:N]

BASE = {"exact_integral": resample_exact_integral, "cubic_spline": resample_cubic_spline,
        "pchip": resample_pchip, "tac_fold": resample_tac_fold}
ARMS = ["original"] + [a for b in BASE for a in (b, b + "_satfix")]
succ = {a: [] for a in ARMS}
sat = {a: [] for a in ARMS}
ep_ids = []
print(f"[{task}] eligible={len(eligible)}/{len(demo_names)} arms={ARMS} k={K}", flush=True)

for idx, dname in enumerate(eligible):
    grp = f["data"][dname]
    a_full = np.asarray(grp["actions"], np.float32)
    L = (len(a_full) // K) * K
    a = np.clip(a_full[:L], -1, 1).astype(np.float32)
    delta, gripper = coarsen_actions(a, K, has_gripper=True)
    nd = delta.shape[1]
    state0 = np.asarray(grp["states"])[0]

    arms = {"original": a}
    for b, fn in BASE.items():
        p = fn(delta, gripper, L)
        arms[b] = p
        arms[b + "_satfix"] = satfix(p, delta, K)

    for arm in ARMS:
        acts = arms[arm]
        assert np.isfinite(acts).all()
        sat[arm].append(float((np.abs(acts[:, :nd]) > 1.0).mean()))
        env.reset()
        env.sim.set_state_from_flattened(state0)
        env.sim.forward()
        s = False
        for act in acts:
            env.step(act)
            if env._check_success():
                s = True
        succ[arm].append(s)
    ep_ids.append(dname)
    if (idx + 1) % 20 == 0 or idx + 1 == len(eligible):
        print(f"[{task}] [{idx+1}/{len(eligible)}] " + " ".join(f"{a}={sum(succ[a])}" for a in ARMS), flush=True)

env.close()
S = {a: np.asarray(succ[a], bool) for a in ARMS}
print(f"\n=== [{task}] k={K} n={len(ep_ids)} ===")
for a in ARMS:
    print(f"  {a:22s} {S[a].mean()*100:6.2f}% ({S[a].sum()}/{len(S[a])})  saturated_elems={100*np.mean(sat[a]):.2f}%")
print("--- paired contrasts vs ZOH (exact_integral) (exact McNemar, 95% CI pp) ---")
res = {"task": task, "k": K, "n": len(ep_ids), "episode_ids": ep_ids,
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
    res["contrasts"][a] = {"delta_pp": delta_pp, "mcnemar_p": p, "ci95_pp": [100 * lo, 100 * hi]}
with open(f"{SD}/robomimic_satfix_{task}_k{K}.json", "w") as fp:
    json.dump(res, fp, indent=2)
print(f"[saved] {SD}/robomimic_satfix_{task}_k{K}.json")
