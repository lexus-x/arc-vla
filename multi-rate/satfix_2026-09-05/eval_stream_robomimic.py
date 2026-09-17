"""
5-Way Stream Comparison on RoboMimic:
TAC-Fold vs BSP vs Cubic Spline vs PCHIP-Fold vs Align-Fold vs Native
No artificial ZOH decimation bottleneck.
Evaluates on real robosuite environments with real demonstration states.
"""
import sys, json, os
import numpy as np
import h5py
import robosuite
from scipy.interpolate import CubicSpline, PchipInterpolator, splprep, splev

DATA_DIR = "/home/user/robomimic_data"
task = sys.argv[1] if len(sys.argv) > 1 else "lift"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2

hdf5_path = f"{DATA_DIR}/{task}.hdf5"
if not os.path.exists(hdf5_path):
    print(f"[{task}] MISSING DATA: {hdf5_path}")
    sys.exit(1)

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
    else:
        delta = reshaped.sum(axis=1)
        return delta, None

def smooth_tac_fold(delta, gripper, k=2):
    n_blocks = len(delta)
    d = delta.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cum = np.vstack([np.zeros((1, d)), np.cumsum(delta, axis=0)])
    slopes = compute_akima_slopes(source_t, cum)
    target_len = n_blocks * k
    resampled_cum = np.empty((target_len + 1, d), dtype=np.float64)
    resampled_cum[0] = cum[0]
    for i in range(n_blocks):
        x0, x1 = source_t[i], source_t[i + 1]
        y0, y1 = cum[i], cum[i + 1]
        d0, d1 = slopes[i], slopes[i + 1]
        sub_t = np.linspace(x0, x1, k + 1)[1:]
        resampled_cum[i * k + 1 : (i + 1) * k + 1] = eval_hermite_cubic(x0, x1, y0, y1, d0, d1, sub_t)
    pose = np.diff(resampled_cum, axis=0).astype(np.float32)
    if gripper is not None:
        source_index = np.minimum(np.floor(np.arange(target_len) / k).astype(int), n_blocks - 1)
        grip = gripper[source_index]
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose

def smooth_cubic_spline(delta, gripper, k=2):
    n_blocks = len(delta)
    d = delta.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cum = np.vstack([np.zeros((1, d)), np.cumsum(delta, axis=0)])
    cs = CubicSpline(source_t, cum, axis=0)
    target_t = np.linspace(0.0, float(n_blocks), n_blocks * k + 1)
    target_cum = cs(target_t)
    pose = np.diff(target_cum, axis=0).astype(np.float32)
    if gripper is not None:
        target_len = n_blocks * k
        source_index = np.minimum(np.floor(np.arange(target_len) / k).astype(int), n_blocks - 1)
        grip = gripper[source_index]
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose

def smooth_pchip_fold(delta, gripper, k=2):
    n_blocks = len(delta)
    d = delta.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cum = np.vstack([np.zeros((1, d)), np.cumsum(delta, axis=0)])
    pchip = PchipInterpolator(source_t, cum, axis=0)
    target_t = np.linspace(0.0, float(n_blocks), n_blocks * k + 1)
    target_cum = pchip(target_t)
    pose = np.diff(target_cum, axis=0).astype(np.float32)
    if gripper is not None:
        target_len = n_blocks * k
        source_index = np.minimum(np.floor(np.arange(target_len) / k).astype(int), n_blocks - 1)
        grip = gripper[source_index]
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose

def smooth_align_fold(delta, gripper, k=2):
    n_blocks = len(delta)
    d = delta.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cum = np.vstack([np.zeros((1, d)), np.cumsum(delta, axis=0)])
    pchip = PchipInterpolator(source_t, cum, axis=0)
    target_t = np.linspace(0.0, float(n_blocks), n_blocks * k + 1)
    target_cum = pchip(target_t)
    pose = np.diff(target_cum, axis=0).astype(np.float32)
    target_len = n_blocks * k
    if gripper is not None:
        grip = np.empty((target_len, 1), dtype=np.float32)
        padded_grip = np.pad(gripper, ((1, 0), (0, 0)), mode="edge")
        for i in range(n_blocks):
            for step in range(k):
                grip[i * k + step] = padded_grip[i + 1] if step == k - 1 else padded_grip[i]
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose

def smooth_b_spline(delta, gripper, k=2, smoothing=0.01):
    n_blocks = len(delta)
    if n_blocks <= 2:
        return np.concatenate([delta, gripper], axis=1) if gripper is not None else delta
    d = delta.shape[1]
    cum = np.vstack([np.zeros((1, d)), np.cumsum(delta, axis=0)])
    target_len = n_blocks * k
    out_cum = np.empty((target_len + 1, d), dtype=np.float32)
    out_cum[0] = cum[0]
    out_cum[-1] = cum[-1]
    t_in = np.linspace(0, 1, len(cum))
    t_out = np.linspace(0, 1, target_len + 1)
    for dim in range(d):
        try:
            tck = splprep([t_in, cum[:, dim]], s=smoothing, k=min(3, len(cum)-1))[0]
            eval_pts = splev(t_out, tck)
            out_cum[:, dim] = eval_pts[1]
        except Exception:
            out_cum[:, dim] = np.interp(t_out, t_in, cum[:, dim])
    pose = np.diff(out_cum, axis=0).astype(np.float32)
    if gripper is not None:
        source_index = np.minimum(np.floor(np.arange(target_len) / k).astype(int), n_blocks - 1)
        grip = gripper[source_index]
        return np.concatenate([pose, grip], axis=1).astype(np.float32)
    return pose

f = h5py.File(hdf5_path, "r")
env_meta = json.loads(f["data"].attrs["env_args"])
env = robosuite.make(env_meta["env_name"], **env_meta["env_kwargs"])

demo_names = list(f["data"].keys())
eligible = [d for d in demo_names if (f["data"][d]["actions"].shape[0] // K) * K >= 16][:N]

ARMS = ["native", "b_spline", "cubic_spline", "pchip_fold", "align_fold", "tac_fold"]
succ = {a: [] for a in ARMS}
sat = {a: [] for a in ARMS}
print(f"[{task}] Starting 5-Way Comparison (n={len(eligible)}/{len(demo_names)}, k={K})", flush=True)

for idx, dname in enumerate(eligible):
    grp = f["data"][dname]
    a_full = np.asarray(grp["actions"], np.float32)
    L = (len(a_full) // K) * K
    a = np.clip(a_full[:L], -1, 1).astype(np.float32)
    delta, gripper = coarsen_actions(a, K, has_gripper=True)
    nd = delta.shape[1]
    state0 = np.asarray(grp["states"])[0]

    arms = {
        "native": a,
        "b_spline": smooth_b_spline(delta, gripper, k=K),
        "cubic_spline": smooth_cubic_spline(delta, gripper, k=K),
        "pchip_fold": smooth_pchip_fold(delta, gripper, k=K),
        "align_fold": smooth_align_fold(delta, gripper, k=K),
        "tac_fold": smooth_tac_fold(delta, gripper, k=K),
    }

    for arm in ARMS:
        acts = arms[arm]
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

    if (idx + 1) % 10 == 0 or idx + 1 == len(eligible):
        status_str = " ".join(f"{a}={sum(succ[a])}" for a in ARMS)
        print(f"  [{task}] [{idx+1:2d}/{len(eligible)}] {status_str}", flush=True)

env.close()

print("\n" + "="*70)
print(f"ROBOMIMIC [{task.upper()}] 5-WAY BENCHMARK RESULTS (n={len(eligible)}, k={K})")
print("="*70)
results = {}
for a in ARMS:
    s_rate = np.mean(succ[a]) * 100
    sat_mean = np.mean(sat[a]) * 100
    print(f"  {a:15s}: {s_rate:5.1f}% ({sum(succ[a])}/{len(succ[a])}) | Saturation: {sat_mean:4.2f}%")
    results[a] = {"success_rate": s_rate, "successes": sum(succ[a]), "total": len(succ[a]), "saturation": sat_mean}

out_path = f"/home/user/Desktop/multi-rate/satfix_2026-09-05/stream_5way_robomimic_{task}_results.json"
with open(out_path, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"[Saved] {out_path}")
