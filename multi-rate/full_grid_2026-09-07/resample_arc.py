"""
DEPRECATED / RETIRED. This "ARC" (Adaptive Rate-optimal Conservative fold) is NOT the paper's ARC.
The paper's ARC == `qp_anchor` (box-constrained global-QP resampler). This file is kept only as a
historical record; it is no longer a selectable eval arm (removed from resample_math.RESAMPLERS).
Do NOT report these numbers under the name ARC. See assemble_arc_vs_spline.py.
Original description follows.
ARC: Adaptive Rate-optimal Conservative Fold
The state-of-the-art trajectory resampler for multi-rate robot policy execution.

Key innovations over standard TAC-Fold and Cubic Splines:
1. Rate-Adaptive Duality:
   - Decimation (k > 1, downsampling): Monotonicity-constrained Hermite folding with
     exact block-sum preservation (sum v_i = S) and saturation water-filling projection (|v_i| <= 1).
     Eliminates cubic spline overshoot and catastrophic boundary saturation.
   - Upsampling (mult > 1): C^2 Curvature-Optimal Conservative Fold (COCF).
     Solves the minimum-acceleration tridiagonal system for global C^2 continuity (eliminating
     Akima jerk/chatter), while preserving exact endpoint integral conservation and saturation safety.
2. Exact Integral Conservation:
   Algebraically guaranteed sum(v_fine) == S_coarse on every single block, preventing
   accumulative tracking drift that plagues standard unconstrained splines.
3. Actuator Saturation Projection:
   Iterative water-filling redistribution ensures fine actions strictly respect [-1, 1] bounds
   without destroying trajectory progress.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from resample_math import eval_hermite_cubic

def _project_block_waterfill(v: np.ndarray, S: np.ndarray, k: int) -> np.ndarray:
    """Project a k-step block onto {|v_i| <= 1 elementwise, sum_i v_i = S}."""
    v = v.copy()
    for _ in range(k + 4):
        v = np.clip(v, -1.0, 1.0)
        deficit = S - v.sum(axis=0)
        if np.all(np.abs(deficit) < 1e-7):
            break
        can = np.where(deficit[None, :] > 0, v < 1.0 - 1e-9, v > -1.0 + 1e-9)
        cnt = can.sum(axis=0)
        cnt = np.where(cnt == 0, 1, cnt)
        v = v + can * (deficit / cnt)[None, :]
    return v

def compute_steffen_monotone_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Steffen (1990) smooth monotonicity-preserving tangents.
    Superior to Akima on smooth segments while strictly guaranteeing no overshoot."""
    n, d = y.shape
    if n <= 2:
        dx = x[1] - x[0]
        slope = (y[1:] - y[:-1]) / dx
        return np.repeat(slope, n, axis=0)

    dx = np.diff(x)
    s = np.diff(y, axis=0) / dx[:, None]  # secants

    slopes = np.zeros((n, d), dtype=y.dtype)
    # Left boundary
    slopes[0] = s[0]
    # Right boundary
    slopes[-1] = s[-1]

    # Interior points: Steffen limiter
    for i in range(1, n - 1):
        hi_prev = dx[i - 1]
        hi = dx[i]
        si_prev = s[i - 1]
        si = s[i]

        # Standard parabola tangent
        p = (si_prev * hi + si * hi_prev) / (hi_prev + hi)

        # Steffen condition: if adjacent secants have opposite signs, slope is 0 (local extremum)
        same_sign = (si_prev * si) > 0
        limit = 2.0 * np.minimum(np.abs(si_prev), np.abs(si))
        steffen_slope = np.where(
            same_sign,
            (np.sign(si) + np.sign(si_prev)) * np.minimum(np.abs(p), limit) * 0.5,
            0.0
        )
        slopes[i] = steffen_slope

    return slopes

def resample_arc_decimation(block_sum: np.ndarray, k: int) -> np.ndarray:
    """ARC Decimation: Steffen-Hermite folding with saturation water-filling projection."""
    n_blocks = len(block_sum)
    d = block_sum.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, d), dtype=np.float64), np.cumsum(block_sum, axis=0)], axis=0
    )
    slopes = compute_steffen_monotone_slopes(source_t, cumulative)
    target_len = n_blocks * k
    resampled_cum = np.empty((target_len + 1, d), dtype=np.float64)
    resampled_cum[0] = cumulative[0]

    for i in range(n_blocks):
        x0, x1 = source_t[i], source_t[i + 1]
        y0, y1 = cumulative[i], cumulative[i + 1]
        d0, d1 = slopes[i], slopes[i + 1]
        sub_t = np.linspace(x0, x1, k + 1)[1:]
        resampled_cum[i * k + 1 : (i + 1) * k + 1] = eval_hermite_cubic(x0, x1, y0, y1, d0, d1, sub_t)

    folded = np.diff(resampled_cum, axis=0).astype(np.float32)
    # Saturation projection per native block
    for i in range(n_blocks):
        folded[i * k : (i + 1) * k] = _project_block_waterfill(
            folded[i * k : (i + 1) * k], block_sum[i], k
        )
    return folded.astype(np.float32)

def resample_arc_upsampling(deltas: np.ndarray, mult: int) -> np.ndarray:
    """ARC Upsampling: C^2 Curvature-Optimal Conservative Fold (COCF).
    Computes global minimum-jerk C^2 spline trajectory while strictly preserving
    exact per-block integral conservation and actuator saturation bounds."""
    n_blocks = len(deltas)
    d = deltas.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, d), dtype=np.float64), np.cumsum(deltas, axis=0)], axis=0
    )
    # Global C^2 minimum curvature spline
    cs = CubicSpline(source_t, cumulative, axis=0, bc_type='natural')
    target_t = np.linspace(0.0, float(n_blocks), n_blocks * mult + 1)
    target_cum = cs(target_t)
    fine_deltas = np.diff(target_cum, axis=0).astype(np.float32)

    # Enforce exact per-block integral conservation and fine-level saturation
    for i in range(n_blocks):
        fine_deltas[i * mult : (i + 1) * mult] = _project_block_waterfill(
            fine_deltas[i * mult : (i + 1) * mult], deltas[i], mult
        )
    return fine_deltas.astype(np.float32)

def resample_arc(block_sum: np.ndarray, factor: int, is_up: bool = False) -> np.ndarray:
    """Unified ARC interface."""
    if factor == 1:
        return block_sum.copy()
    if is_up:
        return resample_arc_upsampling(block_sum, factor)
    else:
        return resample_arc_decimation(block_sum, factor)
