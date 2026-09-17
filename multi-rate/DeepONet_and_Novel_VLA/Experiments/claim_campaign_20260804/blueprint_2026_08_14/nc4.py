"""Step 4a: does actuator/prediction noise widen folding's downsampling advantage?

PRE-REGISTERED before running:
  H1  folding/spline error ratio INCREASES with noise sigma  -> item 4 of the blueprint holds
  H0  ratio flat or DECREASES with sigma                     -> item 4's premise is wrong
Mechanism under test: folding integrates the exact cell integral over dt (averages k
sub-samples, noise ~ sigma/sqrt(k)); spline point-samples an interpolant fitted to noisy
knots (noise ~ sigma). Gain should grow as sqrt(k) with the downsample factor k.
"""
import numpy as np
from scipy.interpolate import CubicSpline

POSE_DIMS = 6
RNG = np.random.default_rng(0)
T, SRC_HZ = 2.5, 20.0
N_SRC = int(T * SRC_HZ)                       # 50-step chunk, the real chunk length

def spline_resample(raw, target_len, native_len):
    """The campaign resampler, raw-space, pose channels only (lines 290-315)."""
    x_s = np.linspace(0.0, 1.0, raw.shape[0])
    x_t = np.linspace(0.0, 1.0, target_len)
    out = CubicSpline(x_s, raw, axis=0)(x_t)
    out *= native_len / target_len            # preserve displacement
    return out

def fold_resample(raw, target_len):
    """Exact cell integral: sum the source deltas falling in each target cell."""
    k = raw.shape[0] // target_len
    return raw[:k * target_len].reshape(target_len, k, -1).sum(axis=1)

def field(t, f):                              # smooth low-freq content, LIBERO-like
    return np.stack([np.sin(2*np.pi*f*t + p) for p in np.linspace(0, 1.2, POSE_DIMS)], -1)

print(f"{'k':>3} {'sigma':>7} {'fold err%':>11} {'spline err%':>12} {'ratio':>9}")
print("-" * 46)
rows = []
for k in (2, 5, 10, 25):                      # 50-step chunk -> 25 / 10 / 5 / 2 steps
    tgt = N_SRC // k
    for sigma in (0.0, 0.01, 0.03, 0.10, 0.30):
        fe, se = [], []
        for _ in range(200):
            t = np.linspace(0, T, N_SRC, endpoint=False)
            dt = T / N_SRC
            clean = field(t + dt/2, 1.0) * dt          # midpoint deltas of the true field
            truth = clean.sum(axis=0)                  # exact total displacement
            noisy = clean + RNG.normal(0, sigma * np.abs(clean).mean(), clean.shape)
            f_out = fold_resample(noisy, tgt).sum(axis=0)
            s_out = spline_resample(noisy, tgt, N_SRC).sum(axis=0)
            fe.append(np.linalg.norm(f_out - truth) / np.linalg.norm(truth))
            se.append(np.linalg.norm(s_out - truth) / np.linalg.norm(truth))
        f_m, s_m = np.mean(fe) * 100, np.mean(se) * 100
        r = s_m / f_m if f_m > 1e-9 else float("inf")
        rows.append((k, sigma, f_m, s_m, r))
        rs = "  EXACT" if r > 1e6 else f"{r:>6.1f}x"
        print(f"{k:>3} {sigma:>7.2f} {f_m:>10.4f}% {s_m:>11.4f}% {rs:>9}")
    print()

print("VERDICT per k -- does the ratio grow with sigma?")
for k in (2, 5, 10, 25):
    rs = [r for kk, s, f, sp, r in rows if kk == k]
    a = "EXACT" if rs[0] > 1e6 else f"{rs[0]:.1f}x"
    print(f"  k={k:<3} sigma 0 -> 0.30 : {a:>7} -> {rs[-1]:>6.1f}x   "
          f"{'H1 (grows)' if rs[-1] > rs[0] else 'H0 REFUTES item 4 (shrinks)'}")
