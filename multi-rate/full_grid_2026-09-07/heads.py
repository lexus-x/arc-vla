"""Action heads for dp_min beyond per-step (PLAN_BLOCKSUM_HEAD.md): targets from a 16-step raw window, decode at speed n.
blocksum: policy predicts nb block MEANS of a 16-step window (exact, lossless); decode = RESAMPLERS[arm](block_sums, kb//n).
bspline : policy predicts C=16 cubic B-spline control points of the cumulative path (fixed clamped uniform knots, LSQ fit);
          decode = evaluate the curve at 16//n+1 points and difference (traverse faster = speedup, as in arXiv 2607.09648)."""
import numpy as np
from scipy.interpolate import BSpline
import resample_math
from resample_qp import resample_qp
resample_math.RESAMPLERS.setdefault("qp", resample_qp)

H = 16          # window (native steps)
C = 16          # B-spline control points
_KNOTS = np.r_[[0.0] * 4, np.linspace(0, H, C - 4 + 2)[1:-1], [float(H)] * 4]  # cubic, clamped, uniform, 12 interior


def _phi(xs):  # design matrix (len(xs), C)
    return BSpline.design_matrix(np.asarray(xs, float), _KNOTS, 3).toarray()


_PHI_FIT = _phi(np.arange(H + 1))                       # 17 x 16
_PHI_PINV = np.linalg.pinv(_PHI_FIT)                    # LSQ fit in one matmul


def targets(ac, head, nd, nb=4):
    """ac (N,16,A) RAW window -> head targets. nd = continuous dims; trailing A-nd dims are causal-hold (gripper etc.)."""
    N, _, A = ac.shape
    if head == "blocksum":
        kb = H // nb
        blk = ac.reshape(N, nb, kb, A)
        return np.concatenate([blk[..., :nd].mean(2), blk[:, :, 0, nd:]], -1).astype(np.float32)   # (N,nb,A)
    if head == "bspline":
        y = np.concatenate([np.zeros((N, 1, nd)), np.cumsum(ac[..., :nd], 1)], 1)                   # (N,17,nd)
        c = np.einsum("ct,ntd->ncd", _PHI_PINV, y)                                                   # (N,16,nd)
        return np.concatenate([c, ac[..., nd:]], -1).astype(np.float32)                              # (N,16,A)
    raise ValueError(head)


def decode(pred, head, nd, n, arm, nb=4):
    """pred = denormalised head output -> executed per-step chunk at speedup n (first half of the horizon, like DP)."""
    if head == "blocksum":
        kb = H // nb; ke = kb // n; assert ke >= 1, f"speed {n} > block size {kb}"
        v = resample_math.RESAMPLERS[arm](pred[:, :nd] * kb, ke)            # block sums -> (nb*ke, nd)
        g = np.repeat(pred[:, nd:], ke, 0)
        return np.concatenate([v, g], 1)[: (nb // 2) * ke].astype(np.float32)
    if head == "bspline":
        y = _phi(np.linspace(0, H, H // n + 1)) @ pred[:, :nd]              # (16/n+1, nd) positions
        return np.concatenate([np.diff(y, axis=0), pred[::n, nd:]], 1)[: H // 2 // n].astype(np.float32)
    raise ValueError(head)


if __name__ == "__main__":  # self-check
    rng = np.random.default_rng(0); ac = np.clip(rng.normal(0, .5, (5, H, 3)), -1, 1); ac[..., 2] = np.sign(ac[..., 2]); nd = 2
    t = targets(ac, "blocksum", nd); full = decode(t[0], "blocksum", nd, 1, "zoh")
    assert full.shape == (8, 3) and np.allclose(full[:, :nd].reshape(2, 4, nd).sum(1), ac[0, :8, :nd].reshape(2, 4, nd).sum(1), atol=1e-6)
    assert np.allclose(full[:, 2], ac[0, :8, 2][[0, 0, 0, 0, 4, 4, 4, 4]])
    for n, arm in ((2, "qp"), (4, "spline_satfix")):
        d = decode(t[0], "blocksum", nd, n, arm); assert d.shape == (8 // n, 3) and np.abs(d[:, :nd]).max() <= 1 + 1e-6
    x = np.arange(H); sm = np.stack([0.8 * np.sin(x / 3), 0.5 * np.cos(x / 4), np.ones(H)], -1)[None]   # smooth path
    tb = targets(sm, "bspline", nd); yb = np.cumsum(decode(tb[0], "bspline", nd, 1, "native")[:, :nd], 0)
    assert tb.shape == (1, 16, 3) and np.abs(yb - np.cumsum(sm[0, :8, :nd], 0)).max() < 1e-4
    assert decode(tb[0], "bspline", nd, 4, "native").shape == (2, 3)
    print("heads self-check ok")
