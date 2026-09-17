"""Step 4a-bis CORRECTED: truth is the ANALYTIC integral, not a sum of the same samples.

  int_0^T sin(2 pi f t + p) dt = (cos(p) - cos(2 pi f T + p)) / (2 pi f)

Previously truth = clean.sum(), which fold_down merely reassociates -> 0% by construction.
Now folding must actually approximate an integral it has not been handed.
"""
import numpy as np
from scipy.interpolate import CubicSpline

POSE_DIMS, T, TGT = 6, 2.5, 50
PH = np.linspace(0, 1.2, POSE_DIMS)

def analytic(f):
    return (np.cos(PH) - np.cos(2*np.pi*f*T + PH)) / (2*np.pi*f)

def spline_down(raw, n):
    x = np.linspace(0.0, 1.0, raw.shape[0])
    return CubicSpline(x, raw, axis=0)(np.linspace(0.0, 1.0, n)) * raw.shape[0] / n

def fold_down(raw, n):
    k = raw.shape[0] // n
    return raw[:k*n].reshape(n, k, -1).sum(axis=1)

for f in (1.0, 5.0):
    truth = analytic(f); nt = np.linalg.norm(truth)
    print(f"\n--- f = {f} Hz, truth = ANALYTIC integral, target fixed at {TGT} steps (20 Hz) ---")
    print(f"{'src Hz':>7} {'k':>4} {'fold err%':>12} {'spline err%':>13} {'ratio':>9}")
    for src_hz in (20, 100, 200, 400, 1000):
        n = int(T*src_hz); k = n // TGT
        dt = T/n
        clean = np.stack([np.sin(2*np.pi*f*(np.linspace(0,T,n,endpoint=False)+dt/2)+p)
                          for p in PH], -1) * dt
        fe = np.linalg.norm(fold_down(clean, TGT).sum(0)-truth)/nt*100
        se = np.linalg.norm(spline_down(clean, TGT).sum(0)-truth)/nt*100
        print(f"{src_hz:>7} {k:>4} {fe:>11.6f}% {se:>12.6f}% {se/fe:>8.1f}x")
