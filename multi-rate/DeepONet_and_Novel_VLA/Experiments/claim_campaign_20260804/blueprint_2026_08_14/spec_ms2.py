"""ManiSkill probe, CORRECTED: action space matched, gripper excluded, dims labelled.

Prior version took a[:, :6] regardless of control mode, so it compared joint-space teleop/mp
against EE-pose-space RL. Here PickCube teleop / mp / rl are ALL pd_joint_delta_pos (8-dim),
so the demonstrator is the only variable. Gripper (last channel) excluded everywhere.
"""
import glob, os, numpy as np, h5py
from scipy.interpolate import CubicSpline
FPS = 20.0

SPACE = {  # name -> (n motion dims, label)
    "pick_teleop": (7, "joint"), "pick_mp": (7, "joint"), "pick_rl_joint": (7, "joint"),
    "pick_rl": (6, "EEpose"), "peg_mp": (7, "joint"), "peg_rl": (7, "joint"),
    "pusht_rl": (6, "EEpose?"), "anymal_rl": (12, "legjoint"),
}

def spline_up(raw, n):
    x = np.linspace(0, 1, raw.shape[0])
    return CubicSpline(x, raw, axis=0)(np.linspace(0, 1, n)) * raw.shape[0] / n

print(f"{'dataset':16s} {'space':9s} {'D':>3s} {'medlen':>7s} {'binHz':>6s} {'f95':>8s} {'>5Hz':>8s} {'spline':>8s} {'n':>4s}")
rows = {}
for f in sorted(glob.glob(os.path.expanduser("~/maniskill_data/*.h5"))):
    name = os.path.basename(f).replace(".h5", "")
    D, lab = SPACE.get(name, (6, "?"))
    f95s, hi5, rt, L = [], [], [], []
    with h5py.File(f, "r") as h:
        for k in [x for x in h.keys() if x.startswith("traj_")][:120]:
            a = np.array(h[k]["actions"]).astype(float)
            if a.ndim != 2 or a.shape[0] < 32: continue
            a = a[:, :D]
            L.append(len(a))
            sig = a - a.mean(0)
            P = (np.abs(np.fft.rfft(sig, axis=0))**2).sum(1); P[0] = 0.0
            if P.sum() <= 0: continue
            fr = np.fft.rfftfreq(len(sig), d=1.0/FPS)
            f95s.append(fr[np.searchsorted(np.cumsum(P)/P.sum(), 0.95)])
            hi5.append(P[fr > 5.0].sum()/P.sum()*100)
            n = (len(a)//2)*2; truth = a[:n].sum(0)
            if np.linalg.norm(truth) < 1e-8: continue
            rt.append(np.linalg.norm(spline_up(a[:n].reshape(n//2,2,D).sum(1), n).sum(0)-truth)
                      / np.linalg.norm(truth)*100)
    ml = np.median(L)
    print(f"{name:16s} {lab:9s} {D:>3d} {ml:>7.0f} {FPS/ml:>6.2f} {np.median(f95s):>7.2f}Hz "
          f"{np.median(hi5):>7.3f}% {np.median(rt):>7.3f}% {len(f95s):>4d}")
