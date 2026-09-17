"""B-spline resamplers implemented the way the B-spline Policy paper (arXiv 2607.09648, Alg. 1)
fits its curves: cubic B-spline, adaptive knot insertion until max pointwise error <= eps, eps in
the data's own units. Fit on the cumulative block-sum path (same convention as spline/pchip/tac_fold
in resample_math.py), evaluated on the fine grid, differenced. Two arms:
  bspline_interp : eps -> 0  (true cubic B-spline INTERPOLANT through every knot, make_interp_spline)
  bspline_eps    : Alg. 1 with eps (least-squares B-spline, knots inserted where error is largest)
Neither uses scipy.splprep's automatic-knot smoothing (`s`), which is what blew up in
resample_math.resample_bspline (830M reconstruction on real data). Per-block sums are NOT preserved
by construction for bspline_eps (that is inherent to a bounded-error fit) -> use the *_satfix
variants for a fair comparison, exactly as for cubic spline / pchip / tac_fold.
"""
import numpy as np
from scipy.interpolate import make_interp_spline, make_lsq_spline, BSpline

def _cum(block_sum):
    d = block_sum.shape[1]
    return np.concatenate([np.zeros((1, d)), np.cumsum(block_sum, axis=0)], axis=0)

def resample_bspline_interp(block_sum, k):
    n = len(block_sum); d = block_sum.shape[1]
    if n < 3:
        return np.repeat(block_sum / k, k, axis=0).astype(np.float32)
    x = np.arange(n + 1, dtype=np.float64); y = _cum(block_sum)
    xs = np.linspace(0.0, float(n), n * k + 1)
    out = np.empty((n * k + 1, d))
    for j in range(d):
        out[:, j] = make_interp_spline(x, y[:, j], k=3)(xs)
    return np.diff(out, axis=0).astype(np.float32)

def _fit_adaptive(x, y, xs, eps, max_knots):
    """Alg. 1 (arXiv 2607.09648): no interior knots -> LSQ cubic fit -> insert one knot in the
    interval containing the max pointwise error -> repeat until max|err| <= eps or budget hit.
    Numerical guards (not in the paper's prose, required for a fair arm): knots at interval
    midpoints (Schoenberg-Whitney safe), interior knots capped at n-5 so the LSQ system stays
    overdetermined, and every candidate fit is validated on the EVALUATION grid xs (finite, within
    the data range +- its span) -- the last valid fit is kept, else the interpolant."""
    n = len(x); yr = float(y.max() - y.min()) + 1e-9; lo, hi = y.min() - yr, y.max() + yr
    interp_vals = make_interp_spline(x, y, k=3)(xs)
    interior = []; best_vals = None
    budget = max(0, min(max_knots, n - 5))
    for _ in range(budget + 1):
        t = np.r_[[x[0]] * 4, interior, [x[-1]] * 4]
        try:
            spl = make_lsq_spline(x, y, t, k=3); fit = spl(x); vals = spl(xs)
        except Exception:
            break
        if not (np.all(np.isfinite(vals)) and vals.min() >= lo and vals.max() <= hi):
            break
        best_vals = vals; err = np.abs(fit - y)
        if err.max() <= eps or len(interior) >= budget:
            break
        i = int(err.argmax())
        # FITPACK-style: bisect the current KNOT SPAN that contains the max-error point, so a
        # persistent local error keeps getting more resolution instead of stalling the loop.
        spans = [x[0]] + interior + [x[-1]]
        a_idx = max(idx for idx, u in enumerate(spans) if u <= x[i] + 1e-12)
        a_idx = min(a_idx, len(spans) - 2)
        cand = 0.5 * (spans[a_idx] + spans[a_idx + 1])
        if any(abs(cand - u) < 1e-9 for u in interior) or (spans[a_idx + 1] - spans[a_idx]) < 0.5:
            break  # span already at data resolution (0.5 = half a block); nothing finer to add
        interior = sorted(interior + [cand])
    _fit_adaptive.last_knots = len(interior)
    return best_vals if best_vals is not None else interp_vals

def resample_bspline_eps(block_sum, k, eps=0.05):
    n = len(block_sum); d = block_sum.shape[1]
    if n < 3:
        return np.repeat(block_sum / k, k, axis=0).astype(np.float32)
    x = np.arange(n + 1, dtype=np.float64); y = _cum(block_sum)
    xs = np.linspace(0.0, float(n), n * k + 1)
    out = np.empty((n * k + 1, d))
    for j in range(d):
        out[:, j] = _fit_adaptive(x, y[:, j], xs, eps, max_knots=max(1, n - 3))
    return np.diff(out, axis=0).astype(np.float32)

if __name__ == "__main__":
    import sys, json, h5py
    sys.path.insert(0, ".")
    from resample_math import coarsen_delta, resample_spline, _satfix_wrap, resample_spline_satfix, resample_zoh, resample_tac_fold_satfix
    from resample_qp import resample_qp
    bi_sf = _satfix_wrap(resample_bspline_interp); be_sf = _satfix_wrap(resample_bspline_eps)
    for task, h5n in (("PushT", "pusht_rl"), ("PickCube", "pick_rl_joint")):
        h5 = h5py.File(f"/home/user/maniskill_data/{h5n}.h5"); meta = json.load(open(f"/home/user/maniskill_data/{h5n}.json"))["episodes"]
        for K in (2, 4):
            mse = {m: [] for m in ("zoh","spline_satfix","tac_fold_satfix","bspline_interp_satfix","bspline_eps_satfix","qp")}
            blow = 0; bserr = []
            for ep in meta[:60]:
                a = np.clip(np.asarray(h5[f"traj_{int(ep['episode_id'])}"]["actions"], np.float32), -1, 1)
                if len(a) < 4*K: continue
                nb = len(a)//K; bs = coarsen_delta(a, K); true = a[:nb*K]
                raw_e = resample_bspline_eps(bs, K); blow += int(np.abs(raw_e).max() > 10)
                bserr.append(float(np.abs(raw_e.reshape(nb,K,-1).sum(1) - bs).mean()))
                for m, fn in (("zoh",resample_zoh),("spline_satfix",resample_spline_satfix),("tac_fold_satfix",resample_tac_fold_satfix),
                              ("bspline_interp_satfix",bi_sf),("bspline_eps_satfix",be_sf),("qp",resample_qp)):
                    r = fn(bs, K); mse[m].append(float(np.mean((r[:nb*K]-true)**2)))
            print(f"{task} k={K} n={len(mse['zoh'])} eps-raw blowups(>10)={blow} eps-raw per-block-sum err={np.mean(bserr):.4f} | " +
                  " ".join(f"{m}={np.mean(v):.5f}" for m, v in mse.items()), flush=True)
        h5.close()
