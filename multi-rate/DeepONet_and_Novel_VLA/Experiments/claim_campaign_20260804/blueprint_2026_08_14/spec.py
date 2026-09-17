"""Option 1 scoping: measure REAL spectral content of demo trajectories.

Maps a dataset onto the measured spline-error curve:  1 Hz content -> 2.75% ceiling (tie),
5 Hz content -> 22.7% (would matter).  Reports f95 (freq below which 95% of AC energy lies)
and the fraction of energy above 5 Hz.  Also runs a direct round-trip: decimate the demo,
rebuild with the campaign spline vs folding, compare cumulative displacement to the truth.
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline

ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/hub")
POSE = 6

def spline_up(raw, n):
    x = np.linspace(0, 1, raw.shape[0])
    return CubicSpline(x, raw, axis=0)(np.linspace(0, 1, n)) * raw.shape[0] / n

for ds in sorted(glob.glob(ROOT + "/datasets--lerobot--libero_*")):
    name = ds.split("--")[-1]
    info = glob.glob(ds + "/snapshots/*/meta/info.json")
    fps = json.load(open(info[0]))["fps"] if info else None
    pqs = sorted(glob.glob(ds + "/snapshots/*/data/**/*.parquet", recursive=True))
    if not pqs: print(f"{name}: no parquet"); continue

    f95s, hi5s, rt_s, rt_f = [], [], [], []
    for pq in pqs[:6]:
        df = pd.read_parquet(pq, columns=["action", "episode_index"])
        for _, g in df.groupby("episode_index"):
            a = np.stack(g["action"].values)[:, :POSE].astype(float)
            if len(a) < 64: continue
            # --- spectrum of the action-delta signal ---
            sig = a - a.mean(0)
            P = (np.abs(np.fft.rfft(sig, axis=0))**2).sum(1)
            fr = np.fft.rfftfreq(len(sig), d=1.0/fps)
            P[0] = 0.0
            c = np.cumsum(P)/P.sum()
            f95s.append(fr[np.searchsorted(c, 0.95)])
            hi5s.append(P[fr > 5.0].sum()/P.sum()*100)
            # --- round trip: decimate x2, rebuild, compare total displacement ---
            n = (len(a)//2)*2
            truth = a[:n].sum(0)
            coarse = a[:n].reshape(n//2, 2, POSE).sum(1)          # exact 2-step integral
            rt_f.append(np.linalg.norm(coarse.sum(0)-truth)/np.linalg.norm(truth)*100)
            rt_s.append(np.linalg.norm(spline_up(coarse, n).sum(0)-truth)/np.linalg.norm(truth)*100)
    print(f"{name:22s} fps={fps:>3}  Nyquist={fps/2:>4.1f} Hz  "
          f"f95={np.median(f95s):>5.2f} Hz  energy>5Hz={np.median(hi5s):>6.3f}%  "
          f"| roundtrip spline={np.median(rt_s):>6.3f}% fold={np.median(rt_f):>6.3f}%  (n_ep={len(f95s)})")
