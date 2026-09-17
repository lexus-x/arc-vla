"""Constrained-smoothing resampler (QP): minimize sum of squared consecutive differences of the
fine-grained velocity sequence, subject to (a) every block's k-step sum equals the true block sum
exactly, (b) every fine-grained value stays in [-1,1]. Solved GLOBALLY across the whole episode in
one convex QP per dimension, not per-block.

Why this is different from everything else in resample_math.py: the spline family (spline, pchip,
tac_fold, bspline) ignores the |v|<=1 box constraint entirely when fitting -- it's applied only as
a post-hoc per-block projection (satfix), which is blind to neighboring blocks: it can only
redistribute slack WITHIN one block, never borrow room from a neighbor. This formulation bakes the
box constraint into the fit itself, jointly across all blocks, so a transition that needs more room
than one block provides can draw on a neighbor's slack via the shared smoothness objective. The
output is already feasible by construction -- no satfix wrap needed.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


def resample_qp(block_sum: np.ndarray, k: int, eps: float = 0.0) -> np.ndarray:
    n_blocks, d = block_sum.shape
    T = n_blocks * k
    out = np.zeros((T, d), dtype=np.float32)
    A = np.zeros((n_blocks, T))
    for i in range(n_blocks):
        A[i, i * k:(i + 1) * k] = 1.0
    bounds = Bounds(-1.0, 1.0)
    C = np.tril(np.ones((n_blocks, n_blocks)))  # cumulative block sums (position at block ends)
    for dim in range(d):
        S = block_sum[:, dim].astype(np.float64)
        S_clipped = np.clip(S, -k, k)  # feasibility requires |S_i|<=k; true block sums always satisfy this
        if eps > 0:  # qp_eps: |position error| <= eps at every block end; at k=1 this is a bounded-error smoother (PREREG_1X)
            lc = LinearConstraint(C @ A, C @ S_clipped - eps, C @ S_clipped + eps)
        else:
            lc = LinearConstraint(A, S_clipped, S_clipped)
        x0 = np.repeat(S_clipped / k, k)  # ZOH warm start -- always feasible

        def obj(v):
            d1 = np.diff(v)
            return float(np.sum(d1 ** 2))

        def grad(v):
            g = np.zeros_like(v)
            d1 = np.diff(v)
            g[:-1] -= 2 * d1
            g[1:] += 2 * d1
            return g

        res = minimize(obj, x0, jac=grad, method="SLSQP", bounds=bounds, constraints=[lc],
                        options={"maxiter": 200, "ftol": 1e-9})
        v = res.x if res.success else x0
        out[:, dim] = v.astype(np.float32)
    return out


if __name__ == "__main__":  # smoke: feasibility + exact block-sum preservation
    rng = np.random.default_rng(0)
    bs = rng.uniform(-1.6, 1.6, size=(20, 3))
    out = resample_qp(bs, 2)
    assert out.shape == (40, 3)
    assert np.abs(out).max() <= 1.0 + 1e-6, out.max()
    recon = out.reshape(20, 2, 3).sum(1)
    assert np.allclose(recon, np.clip(bs, -2, 2), atol=1e-5), np.abs(recon - np.clip(bs, -2, 2)).max()
    print("resample_qp smoke OK: feasible and block-sum-exact")


def resample_hybrid(block_sum: np.ndarray, k: int, base_resampler=None, base_satfix=None) -> np.ndarray:
    """Route only ISOLATED violating blocks (base interpolant's raw output exceeds |v|>1, but
    neither neighbor does) through the global QP; everywhere else, use the base interpolant's
    already-satfixed output unchanged. Detectable at resample time from the base interpolant's own
    raw output alone -- no oracle/future information needed, so this is actually deployable, unlike
    the block-sum-feature selectors that already failed in the 2026-09-06 diagnostic (those needed
    information the resampler doesn't have; this needs only what it already computes)."""
    from resample_math import resample_spline, resample_spline_satfix
    base_resampler = base_resampler or resample_spline
    base_satfix = base_satfix or resample_spline_satfix
    n_blocks, d = block_sum.shape
    raw = base_resampler(block_sum, k)
    fixed = base_satfix(block_sum, k)
    raw_blocks = raw[:n_blocks * k].reshape(n_blocks, k, d)
    active = (np.abs(raw_blocks) > 1.0).any(axis=1)  # (n_blocks, d)
    isolated = np.zeros_like(active)
    if n_blocks == 1:
        isolated[0] = active[0]
    else:
        isolated[1:-1] = active[1:-1] & ~active[:-2] & ~active[2:]
        isolated[0] = active[0] & ~active[1]
        isolated[-1] = active[-1] & ~active[-2]
    if not isolated.any():
        return fixed
    qp_full = resample_qp(block_sum, k)
    out = fixed.copy()
    qp_blocks = qp_full[:n_blocks * k].reshape(n_blocks, k, d)
    out_blocks = out[:n_blocks * k].reshape(n_blocks, k, d)
    mask = np.broadcast_to(isolated[:, None, :], qp_blocks.shape)
    out_blocks[mask] = qp_blocks[mask]
    return out


if __name__ == "__main__":  # self-check
    rng = np.random.default_rng(0); c = np.clip(rng.normal(0, .4, (8, 2)), -1, 1)
    assert np.allclose(resample_qp(c, 1), c, atol=1e-5)  # eps=0 at k=1 is identity
    v = resample_qp(c, 1, eps=0.05)
    assert np.abs(np.cumsum(v - c, 0)).max() <= 0.05 + 1e-6 and np.abs(v).max() <= 1 + 1e-9
    assert np.sum(np.diff(v, axis=0) ** 2) < np.sum(np.diff(c, axis=0) ** 2)  # smoother than input
    print("resample_qp self-check ok")


def resample_qp_anchor(block_sum: np.ndarray, k: int, anchor_fn=None) -> np.ndarray:
    """Same global QP as resample_qp, but the smoothness objective is anchored to an anchor
    curve's own shape instead of to flatness: minimize sum_t ((v_{t+1}-v_t)-(a_{t+1}-a_t))^2,
    same box/block-sum constraints. resample_qp is the special case anchor=0.

    Motivation: plain resample_qp has no data term, so absent a box violation it drifts toward
    whatever curve minimizes raw consecutive-difference energy -- shape-blind, which is the
    diagnosed reason it loses to TAC-Fold at low decimation (TAC-Fold's Akima-damped curve
    already tracks the true within-block dynamics; QP's flatness prior fights that for nothing
    when there's no saturation to fix). The default anchor, TAC-Fold, satisfies the block-sum
    constraint exactly by construction (interpolates through every cumulative block boundary),
    so wherever TAC-Fold's own raw output is already inside [-1,1], v=anchor is a zero-cost
    global optimum and this reproduces TAC-Fold exactly. Wherever the anchor violates the box,
    the same cross-block coupling as resample_qp lets a neighboring block absorb the deficit
    while staying as close as possible to the anchor's shape everywhere else.
    """
    from resample_math import resample_tac_fold
    anchor_fn = anchor_fn or resample_tac_fold
    anchor = anchor_fn(block_sum, k).astype(np.float64)
    n_blocks, d = block_sum.shape
    T = n_blocks * k
    out = np.zeros((T, d), dtype=np.float32)
    A = np.zeros((n_blocks, T))
    for i in range(n_blocks):
        A[i, i * k:(i + 1) * k] = 1.0
    bounds = Bounds(-1.0, 1.0)
    for dim in range(d):
        S = block_sum[:, dim].astype(np.float64)
        S_clipped = np.clip(S, -k, k)
        lc = LinearConstraint(A, S_clipped, S_clipped)
        a = anchor[:, dim]
        x0 = np.repeat(S_clipped / k, k)  # ZOH warm start -- always feasible

        def obj(v, a=a):
            d1 = np.diff(v) - np.diff(a)
            return float(np.sum(d1 ** 2))

        def grad(v, a=a):
            g = np.zeros_like(v)
            d1 = np.diff(v) - np.diff(a)
            g[:-1] -= 2 * d1
            g[1:] += 2 * d1
            return g

        res = minimize(obj, x0, jac=grad, method="SLSQP", bounds=bounds, constraints=[lc],
                        options={"maxiter": 200, "ftol": 1e-9})
        v = res.x if res.success else x0
        out[:, dim] = v.astype(np.float32)
    return out


if __name__ == "__main__":  # self-check: exact TAC-Fold reproduction when TAC-Fold is feasible
    from resample_math import resample_tac_fold
    rng = np.random.default_rng(1)
    bs = np.clip(rng.normal(0, 0.3, (12, 2)), -0.9, 0.9)  # small sums -> TAC-Fold stays in-box
    anchored = resample_qp_anchor(bs, 2)
    tac = resample_tac_fold(bs, 2)
    assert np.abs(anchored).max() <= 1.0 + 1e-6
    assert np.allclose(anchored, tac, atol=1e-4), np.abs(anchored - tac).max()
    bs_sat = rng.uniform(-1.8, 1.8, size=(12, 2))  # forces TAC-Fold to violate the box somewhere
    anchored2 = resample_qp_anchor(bs_sat, 2)
    assert np.abs(anchored2).max() <= 1.0 + 1e-6
    recon = anchored2.reshape(12, 2, 2).sum(1)
    assert np.allclose(recon, np.clip(bs_sat, -2, 2), atol=1e-5)
    print("resample_qp_anchor self-check ok: reproduces TAC-Fold when feasible, else feasible+coupled")
