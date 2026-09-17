"""ManiSkill3 probe: do NON-teleop demos (motion planning / RL) break the ~2 Hz teleop ceiling?
PickCube-v1 has teleop + motionplanning + rl of the SAME task -> controlled comparison.
AnymalC-Reach (quadruped locomotion, RL) is the high-frequency positive control."""
import glob, os, json, numpy as np, h5py
from scipy.interpolate import CubicSpline

def spline_up(raw, n):
    x = np.linspace(0, 1, raw.shape[0])
    return CubicSpline(x, raw, axis=0)(np.linspace(0, 1, n)) * raw.shape[0] / n

print(f"{'dataset':14s} {'src':>6s} {'fps':>5s} {'Nyq':>5s} {'f95':>8s} {'>5Hz':>8s} {'spline':>8s} {'n_ep':>5s}")
for f in sorted(glob.glob(os.path.expanduser("~/maniskill_data/*.h5"))):
    name = os.path.basename(f).replace(".h5", "")
    jf = f.replace(".h5", ".json")
    fps, src = 20.0, name.split("_")[-1]
    if os.path.exists(jf):
        try:
            j = json.load(open(jf))
            ei = j.get("env_info", {}).get("env_kwargs", {})
            fps = float(ei.get("control_freq", j.get("env_info", {}).get("control_freq", 20)))
        except Exception: pass
    f95s, hi5, rt_s = [], [], []
    try: h = h5py.File(f, "r")
    except Exception as e: print(f"{name:14s} UNREADABLE {e}"); continue
    with h:
        for k in list(h.keys())[:60]:
            try: a = np.array(h[k]["actions"]).astype(float)
            except Exception: continue
            if a.ndim != 2 or len(a) < 32: continue
            D = min(a.shape[1], 6); a = a[:, :D]
            sig = a - a.mean(0)
            P = (np.abs(np.fft.rfft(sig, axis=0))**2).sum(1); P[0] = 0.0
            if P.sum() <= 0: continue
            fr = np.fft.rfftfreq(len(sig), d=1.0/fps)
            f95s.append(fr[np.searchsorted(np.cumsum(P)/P.sum(), 0.95)])
            hi5.append(P[fr > 5.0].sum()/P.sum()*100)
            n = (len(a)//2)*2; truth = a[:n].sum(0)
            if np.linalg.norm(truth) < 1e-8: continue
            coarse = a[:n].reshape(n//2, 2, D).sum(1)
            rt_s.append(np.linalg.norm(spline_up(coarse, n).sum(0)-truth)/np.linalg.norm(truth)*100)
    if not f95s: print(f"{name:14s} no usable episodes"); continue
    print(f"{name:14s} {src:>6s} {fps:>5.0f} {fps/2:>5.1f} {np.median(f95s):>7.2f}Hz "
          f"{np.median(hi5):>7.3f}% {np.median(rt_s) if rt_s else float('nan'):>7.3f}% {len(f95s):>5d}")
