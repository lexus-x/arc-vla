"""
5-Way Stream Comparison: TAC-Fold vs BSP vs Cubic Spline vs PCHIP-Fold vs Align-Fold vs Native
Directly addresses user question:
1. No ZOH decimation bottleneck -- smoothers applied directly to action trajectory.
2. Head-to-head comparison of:
   - native: raw un-smoothed actions
   - b_spline: B-spline Policy (BSP) approximating curve with convex hull smoothing
   - cubic_spline: global C2 cubic spline
   - pchip_fold: Piecewise Cubic Hermite Interpolating Polynomial (monotonic)
   - align_fold: Phase-aligned Hermite with endpoint gripper sync
   - tac_fold: Taut-Akima Conservative Fold (Akima slope-limited C1 Hermite)
"""
import sys, json, os
import numpy as np
import h5py
from scipy.interpolate import CubicSpline, PchipInterpolator, splprep, splev

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

def smooth_tac_fold(delta, k=2):
    # k-step block Akima folding
    n_blocks = len(delta) // k
    if n_blocks <= 1:
        return delta.copy()
    block_sum = delta[:n_blocks*k].reshape(n_blocks, k, delta.shape[1]).sum(axis=1)
    d = block_sum.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cum = np.vstack([np.zeros((1, d)), np.cumsum(block_sum, axis=0)])
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
    out = np.diff(resampled_cum, axis=0).astype(np.float32)
    if len(delta) > target_len:
        out = np.vstack([out, delta[target_len:]])
    return out

def smooth_cubic_spline(delta, k=2):
    n_blocks = len(delta) // k
    if n_blocks <= 1:
        return delta.copy()
    block_sum = delta[:n_blocks*k].reshape(n_blocks, k, delta.shape[1]).sum(axis=1)
    d = block_sum.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cum = np.vstack([np.zeros((1, d)), np.cumsum(block_sum, axis=0)])
    cs = CubicSpline(source_t, cum, axis=0)
    target_t = np.linspace(0.0, float(n_blocks), n_blocks * k + 1)
    target_cum = cs(target_t)
    out = np.diff(target_cum, axis=0).astype(np.float32)
    if len(delta) > n_blocks * k:
        out = np.vstack([out, delta[n_blocks*k:]])
    return out

def smooth_pchip_fold(delta, k=2):
    n_blocks = len(delta) // k
    if n_blocks <= 1:
        return delta.copy()
    block_sum = delta[:n_blocks*k].reshape(n_blocks, k, delta.shape[1]).sum(axis=1)
    d = block_sum.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cum = np.vstack([np.zeros((1, d)), np.cumsum(block_sum, axis=0)])
    pchip = PchipInterpolator(source_t, cum, axis=0)
    target_t = np.linspace(0.0, float(n_blocks), n_blocks * k + 1)
    target_cum = pchip(target_t)
    out = np.diff(target_cum, axis=0).astype(np.float32)
    if len(delta) > n_blocks * k:
        out = np.vstack([out, delta[n_blocks*k:]])
    return out

def smooth_b_spline(delta, k=2, smoothing=0.01):
    # Approximating B-Spline (BSP style): fits a continuous B-spline curve with convex hull smoothing
    n_blocks = len(delta) // k
    if n_blocks <= 2:
        return delta.copy()
    block_sum = delta[:n_blocks*k].reshape(n_blocks, k, delta.shape[1]).sum(axis=1)
    d = block_sum.shape[1]
    cum = np.vstack([np.zeros((1, d)), np.cumsum(block_sum, axis=0)])
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
            # Fallback to linear if splprep encounters degenerate variance
            out_cum[:, dim] = np.interp(t_out, t_in, cum[:, dim])
    out = np.diff(out_cum, axis=0).astype(np.float32)
    if len(delta) > target_len:
        out = np.vstack([out, delta[target_len:]])
    return out

def smooth_align_fold(delta, gripper=None, k=2):
    pose = smooth_pchip_fold(delta, k=k)
    if gripper is not None:
        target_len = len(pose)
        source_len = len(gripper) // k
        ratio = k
        grip = np.empty((target_len, 1), dtype=np.float32)
        padded_grip = np.pad(gripper[:source_len*k:k], ((1, 0), (0, 0)), mode="edge")
        for i in range(source_len):
            for step in range(ratio):
                # hold until arrival at boundary
                grip[i * ratio + step] = padded_grip[i + 1] if step == ratio - 1 else padded_grip[i]
        return pose, grip
    return pose, None

def run_pusht_eval(n_episodes=50, k=2):
    import gymnasium as gym, mani_skill.envs
    import torch
    H5 = "/home/user/maniskill_data/pusht_rl.h5"
    JS = "/home/user/maniskill_data/pusht_rl.json"
    meta = json.loads(open(JS).read())
    with h5py.File(H5) as h:
        eligible = [e for e in meta["episodes"] if len(h[f"traj_{int(e['episode_id'])}"]["actions"]) >= 16][:n_episodes]

    env = gym.make("PushT-v1", num_envs=1, obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="physx_cuda", reconfiguration_freq=1)
    u = env.unwrapped
    
    ARMS = ["native", "b_spline", "cubic_spline", "pchip_fold", "align_fold", "tac_fold"]
    succ = {a: [] for a in ARMS}
    sat = {a: [] for a in ARMS}
    print(f"\n[PushT] Starting 5-Way Comparison (n={len(eligible)}, k={k})", flush=True)

    with h5py.File(H5) as h:
        for idx, ep in enumerate(eligible):
            ep_id = int(ep["episode_id"])
            a_full = np.asarray(h[f"traj_{ep_id}"]["actions"], np.float32)
            L = (len(a_full) // k) * k
            a = np.clip(a_full[:L], -1, 1).astype(np.float32)
            
            # Generate smoothed trajectories
            arms_actions = {
                "native": a,
                "b_spline": smooth_b_spline(a, k=k),
                "cubic_spline": smooth_cubic_spline(a, k=k),
                "pchip_fold": smooth_pchip_fold(a, k=k),
                "align_fold": smooth_align_fold(a, k=k)[0],
                "tac_fold": smooth_tac_fold(a, k=k),
            }

            sg = h[f"traj_{ep_id}"]["env_states"]
            state = {g: {n: torch.as_tensor(np.asarray(sg[g][n])[0:1]) for n in sg[g]} for g in sg}

            for arm in ARMS:
                acts = arms_actions[arm]
                sat[arm].append(float((np.abs(acts) > 1.0).mean()))
                env.reset(seed=ep["episode_seed"])
                u.set_state_dict(state)
                s = False
                for act in acts:
                    _, _, term, trunc, info = env.step(act[None])
                    succ_tensor = info.get("success")
                    if succ_tensor is not None:
                        if hasattr(succ_tensor, "cpu"):
                            s_val = bool(succ_tensor.cpu().numpy().reshape(-1)[0])
                        else:
                            s_val = bool(np.asarray(succ_tensor).reshape(-1)[0])
                        if s_val:
                            s = True
                    
                    term_val = term.cpu().numpy().reshape(-1)[0] if hasattr(term, "cpu") else np.asarray(term).reshape(-1)[0]
                    trunc_val = trunc.cpu().numpy().reshape(-1)[0] if hasattr(trunc, "cpu") else np.asarray(trunc).reshape(-1)[0]
                    if bool(term_val) or bool(trunc_val):
                        break
                succ[arm].append(s)

            if (idx + 1) % 10 == 0 or idx + 1 == len(eligible):
                status_str = " ".join(f"{a}={sum(succ[a])}" for a in ARMS)
                print(f"  [{idx+1:2d}/{len(eligible)}] {status_str}", flush=True)

    env.close()
    
    print("\n" + "="*70)
    print(f"PUSHT 5-WAY BENCHMARK RESULTS (n={len(eligible)}, k={k})")
    print("="*70)
    results = {}
    for a in ARMS:
        s_rate = np.mean(succ[a]) * 100
        sat_mean = np.mean(sat[a]) * 100
        print(f"  {a:15s}: {s_rate:5.1f}% ({sum(succ[a])}/{len(succ[a])}) | Saturation: {sat_mean:4.2f}%")
        results[a] = {"success_rate": s_rate, "successes": sum(succ[a]), "total": len(succ[a]), "saturation": sat_mean}
    
    out_path = "/home/user/Desktop/multi-rate/satfix_2026-09-05/stream_5way_pusht_results.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[Saved] {out_path}")
    return results

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_pusht_eval(n, k)
