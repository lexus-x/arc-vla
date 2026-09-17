"""Same screen, robomimic HDF5. robosuite records at 20 Hz -> Nyquist 10 Hz, so unlike
LIBERO these datasets CAN represent 5 Hz content. Question is whether they do."""
import glob, os
import numpy as np, h5py
from scipy.interpolate import CubicSpline

FPS, POSE = 20.0, 6      # robosuite control_freq=20 for the rt_benchmark sets

def spline_up(raw, n):
    x = np.linspace(0, 1, raw.shape[0])
    return CubicSpline(x, raw, axis=0)(np.linspace(0, 1, n)) * raw.shape[0] / n

print(f"{'dataset':12s} {'fps':>4} {'Nyq':>5} {'f95':>7} {'>5Hz':>8} {'spline':>8} {'fold':>7} {'n_ep':>5}")
for f in sorted(glob.glob(os.path.expanduser("~/robomimic_data/*.hdf5"))):
    name = os.path.basename(f).replace(".hdf5", "")
    f95s, hi5, rt_s, rt_f = [], [], [], []
    with h5py.File(f, "r") as h:
        eps = list(h["data"].keys())
        for k in eps[:40]:
            a = np.array(h[f"data/{k}/actions"])[:, :POSE].astype(float)
            if len(a) < 64: continue
            sig = a - a.mean(0)
            P = (np.abs(np.fft.rfft(sig, axis=0))**2).sum(1); P[0] = 0.0
            fr = np.fft.rfftfreq(len(sig), d=1.0/FPS)
            f95s.append(fr[np.searchsorted(np.cumsum(P)/P.sum(), 0.95)])
            hi5.append(P[fr > 5.0].sum()/P.sum()*100)
            n = (len(a)//2)*2
            truth = a[:n].sum(0)
            coarse = a[:n].reshape(n//2, 2, POSE).sum(1)
            rt_f.append(np.linalg.norm(coarse.sum(0)-truth)/np.linalg.norm(truth)*100)
            rt_s.append(np.linalg.norm(spline_up(coarse, n).sum(0)-truth)/np.linalg.norm(truth)*100)
    print(f"{name:12s} {FPS:>4.0f} {FPS/2:>5.1f} {np.median(f95s):>6.2f}Hz "
          f"{np.median(hi5):>7.3f}% {np.median(rt_s):>7.3f}% {np.median(rt_f):>6.3f}% {len(f95s):>5}")
