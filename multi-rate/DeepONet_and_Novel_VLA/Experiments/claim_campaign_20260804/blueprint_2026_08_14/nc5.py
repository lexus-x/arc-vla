"""Step 4a-bis: the ACTUAL proposed experiment -- sub-native query, native execution.

Target length FIXED at 50 (real 20 Hz execution of a 2.5 s chunk).
SOURCE density varies: the operator is queried at 20/100/200/400 Hz and integrated down.
Spline must interpolate the same dense source down to 50.

PRE-REGISTERED: if spline error stays < ~1% as source density grows, sub-native query
buys nothing and blueprint item 4 is dead alongside its noise premise.
"""
import numpy as np
from scipy.interpolate import CubicSpline

POSE_DIMS, T, TGT = 6, 2.5, 50          # 50 steps @ 20 Hz = native execution

def spline_down(raw, target_len):
    x_s = np.linspace(0.0, 1.0, raw.shape[0])
    out = CubicSpline(x_s, raw, axis=0)(np.linspace(0.0, 1.0, target_len))
    return out * raw.shape[0] / target_len

def fold_down(raw, target_len):
    k = raw.shape[0] // target_len
    return raw[:k*target_len].reshape(target_len, k, -1).sum(axis=1)

def field(t, f):
    return np.stack([np.sin(2*np.pi*f*t + p) for p in np.linspace(0, 1.2, POSE_DIMS)], -1)

for f in (1.0, 5.0):                     # 1 Hz = LIBERO-like content; 5 Hz = dynamic
    print(f"\n--- content frequency f = {f} Hz  (target fixed at {TGT} steps = 20 Hz) ---")
    print(f"{'src Hz':>7} {'k':>4} {'fold err%':>11} {'spline err%':>12} {'ratio':>9}")
    for src_hz in (20, 100, 200, 400):
        n = int(T*src_hz); k = n // TGT
        t  = np.linspace(0, T, n, endpoint=False); dt = T/n
        clean = field(t + dt/2, f) * dt
        truth = clean.sum(axis=0)
        fe = np.linalg.norm(fold_down(clean, TGT).sum(0)-truth)/np.linalg.norm(truth)*100
        se = np.linalg.norm(spline_down(clean, TGT).sum(0)-truth)/np.linalg.norm(truth)*100
        rr = "  EXACT" if fe < 1e-9 else f"{se/fe:>6.1f}x"
        print(f"{src_hz:>7} {k:>4} {fe:>10.6f}% {se:>11.6f}% {rr:>9}")
