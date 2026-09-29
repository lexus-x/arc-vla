"""
Decimation + resampling + satfix math, ported verbatim from the validated ManiSkill/LIBERO
harness so RoboCasa gets the exact same mechanism, not a reimplementation:
  - compute_akima_slopes / eval_hermite_cubic / resample_tac_fold / resample_cubic_spline:
    /home/user/Desktop/multi-rate/vla-vault/scratch/sweep_maniskill_decimation_ratios.py
  - satfix block projection:
    /home/user/Desktop/multi-rate/satfix_2026-09-05/run_satfix_general.py
    and its k-generic form in
    /tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad/run_satfix_libero_v2flow.py
"""
import numpy as np
from scipy.interpolate import CubicSpline


def compute_akima_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, d = y.shape
    if n <= 2:
        dx = x[1] - x[0]
        slope = (y[1:] - y[:-1]) / dx
        return np.repeat(slope, n, axis=0)

    dx = np.diff(x)
    m = np.diff(y, axis=0) / dx[:, None]

    m_pad = np.empty((n + 3, d), dtype=y.dtype)
    m_pad[2:-2] = m
    m_pad[1] = 2.0 * m[0] - m[1]
    m_pad[0] = 2.0 * m_pad[1] - m[0]
    m_pad[-2] = 2.0 * m[-1] - m[-2]
    m_pad[-1] = 2.0 * m_pad[-2] - m[-1]

    dm = np.abs(np.diff(m_pad, axis=0))
    w1 = dm[2:]
    w2 = dm[:-2]

    weights_sum = w1 + w2
    zero_mask = weights_sum < 1e-12
    weights_sum_safe = np.where(zero_mask, 1.0, weights_sum)

    slopes = (w1 * m_pad[1:-2] + w2 * m_pad[2:-1]) / weights_sum_safe
    slopes = np.where(zero_mask, 0.5 * (m_pad[1:-2] + m_pad[2:-1]), slopes)
    return slopes


def eval_hermite_cubic(x0, x1, y0, y1, d0, d1, x):
    h = x1 - x0
    t = (x - x0) / h
    t2 = t * t
    t3 = t2 * t

    h00 = (2.0 * t3 - 3.0 * t2 + 1.0)[:, None]
    h10 = (t3 - 2.0 * t2 + t)[:, None] * h
    h01 = (-2.0 * t3 + 3.0 * t2)[:, None]
    h11 = (t3 - t2)[:, None] * h

    return h00 * y0[None, :] + h10 * d0[None, :] + h01 * y1[None, :] + h11 * d1[None, :]


def coarsen_delta(delta_steps: np.ndarray, k: int) -> np.ndarray:
    """Sum every block of k native steps -> per-block total displacement S."""
    n_blocks = len(delta_steps) // k
    truncated = delta_steps[: n_blocks * k]
    return truncated.reshape(n_blocks, k, delta_steps.shape[1]).sum(axis=1)


def coarsen_gripper_transitions(raw_col: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Record each complete block's final-run offset and ending command.

    The offset is the first index in the maximal trailing run equal to the block's
    final command, or ``-1`` when the final command equals the starting command.
    The parallel value array keeps reconstruction independent of assumptions about
    the command encoding.
    """
    raw_col = np.asarray(raw_col)
    if raw_col.ndim != 1:
        raise ValueError(f"raw_col must have shape (T,), got {raw_col.shape}")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")

    n_blocks = len(raw_col) // k
    blocks = raw_col[: n_blocks * k].reshape(n_blocks, k)
    offsets = np.full(n_blocks, -1, dtype=np.int64)
    end_values = blocks[:, -1].copy()
    for block_index, block in enumerate(blocks):
        if block[-1] == block[0]:
            continue
        run_start = k - 1
        while run_start > 0 and block[run_start - 1] == block[-1]:
            run_start -= 1
        offsets[block_index] = run_start
    return offsets, end_values


def resample_zoh(block_sum: np.ndarray, k: int) -> np.ndarray:
    """Equal-split ZOH: S/k repeated k times per block."""
    return np.repeat(block_sum / k, k, axis=0).astype(np.float32)


def resample_tac_fold(block_sum: np.ndarray, k: int) -> np.ndarray:
    """TAC-Fold (Taut-Akima Conservative Fold) on the cumulative trajectory of block sums."""
    n_blocks = len(block_sum)
    d = block_sum.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, d), dtype=np.float64), np.cumsum(block_sum, axis=0)], axis=0
    )
    slopes = compute_akima_slopes(source_t, cumulative)
    target_len = n_blocks * k
    resampled_cum = np.empty((target_len + 1, d), dtype=np.float64)
    resampled_cum[0] = cumulative[0]
    for i in range(n_blocks):
        x0, x1 = source_t[i], source_t[i + 1]
        y0, y1 = cumulative[i], cumulative[i + 1]
        d0, d1 = slopes[i], slopes[i + 1]
        sub_t = np.linspace(x0, x1, k + 1)[1:]
        resampled_cum[i * k + 1 : (i + 1) * k + 1] = eval_hermite_cubic(x0, x1, y0, y1, d0, d1, sub_t)
    return np.diff(resampled_cum, axis=0).astype(np.float32)


def resample_spline(block_sum: np.ndarray, k: int) -> np.ndarray:
    """Cubic spline (unconstrained, not Akima-damped) on the cumulative trajectory of block
    sums -- the arm that originally lost to ZOH in the ManiSkill open-loop work because
    plain cubic splines overshoot |1| far more than TAC-Fold's slope-limited Hermite does."""
    n_blocks = len(block_sum)
    d = block_sum.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, d), dtype=np.float64), np.cumsum(block_sum, axis=0)], axis=0
    )
    cs = CubicSpline(source_t, cumulative, axis=0)
    target_t = np.linspace(0.0, float(n_blocks), n_blocks * k + 1)
    target_cum = cs(target_t)
    return np.diff(target_cum, axis=0).astype(np.float32)


def _project_block(v: np.ndarray, S: np.ndarray, k: int) -> np.ndarray:
    """Project a k-step block onto {|v_i|<=1 elementwise, sum_i v_i = S}.

    Clip then redistribute the deficit equally over elements that can still move in the
    needed direction, repeat. Feasible whenever |S|<=k (guaranteed since S is itself a sum
    of k already-clipped native actions).
    """
    v = v.copy()
    for _ in range(k + 2):
        v = np.clip(v, -1.0, 1.0)
        deficit = S - v.sum(0)
        if np.all(np.abs(deficit) < 1e-7):
            break
        can = np.where(deficit[None, :] > 0, v < 1.0 - 1e-9, v > -1.0 + 1e-9)
        cnt = can.sum(0)
        cnt = np.where(cnt == 0, 1, cnt)
        v = v + can * (deficit / cnt)[None, :]
    return v


def _satfix_wrap(base_fn):
    def wrapped(block_sum: np.ndarray, k: int) -> np.ndarray:
        folded = base_fn(block_sum, k)
        n_blocks = len(block_sum)
        for i in range(n_blocks):
            folded[i * k : (i + 1) * k] = _project_block(folded[i * k : (i + 1) * k], block_sum[i], k)
        return folded.astype(np.float32)
    return wrapped


resample_tac_fold_satfix = _satfix_wrap(resample_tac_fold)
resample_spline_satfix = _satfix_wrap(resample_spline)


def resample_pchip(block_sum: np.ndarray, k: int) -> np.ndarray:
    """PCHIP (Fritsch-Carlson monotone Hermite) on the cumulative block-sum trajectory."""
    from scipy.interpolate import PchipInterpolator
    n_blocks = len(block_sum)
    d = block_sum.shape[1]
    source_t = np.arange(n_blocks + 1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((1, d), dtype=np.float64), np.cumsum(block_sum, axis=0)], axis=0
    )
    target_t = np.linspace(0.0, float(n_blocks), n_blocks * k + 1)
    return np.diff(PchipInterpolator(source_t, cumulative, axis=0)(target_t), axis=0).astype(np.float32)


resample_pchip_satfix = _satfix_wrap(resample_pchip)


def resample_bspline(block_sum: np.ndarray, k: int, smoothing: float = 0.01) -> np.ndarray:
    """Smoothing B-spline (scipy splprep/splev) on the cumulative block-sum trajectory --
    ported from satfix_2026-09-05/eval_stream_robomimic.py:smooth_b_spline, adapted to the
    (block_sum, k) -> resampled-deltas interface shared by spline/pchip/tac_fold."""
    from scipy.interpolate import splprep, splev
    n_blocks = len(block_sum)
    d = block_sum.shape[1]
    if n_blocks <= 2:
        return resample_zoh(block_sum, k)
    cumulative = np.concatenate(
        [np.zeros((1, d), dtype=np.float64), np.cumsum(block_sum, axis=0)], axis=0
    )
    t_in = np.linspace(0.0, 1.0, n_blocks + 1)
    t_out = np.linspace(0.0, 1.0, n_blocks * k + 1)
    out_cum = np.empty((n_blocks * k + 1, d), dtype=np.float64)
    out_cum[0], out_cum[-1] = cumulative[0], cumulative[-1]
    for dim in range(d):
        try:
            tck, _ = splprep([t_in, cumulative[:, dim]], s=smoothing, k=min(3, n_blocks - 1))
            out_cum[:, dim] = splev(t_out, tck)[1]
        except Exception:
            out_cum[:, dim] = np.interp(t_out, t_in, cumulative[:, dim])
    return np.diff(out_cum, axis=0).astype(np.float32)


resample_bspline_satfix = _satfix_wrap(resample_bspline)


GRIPPER_SYNC_CONTINUOUS_RESAMPLER = "spline_satfix"


def gripper_sync(
    chunk: np.ndarray,
    transition_summary: tuple[np.ndarray, np.ndarray],
    n_hold: int,
    k: int,
) -> np.ndarray:
    """Reconstruct gripper switches from one offset/value summary per block.

    ``chunk`` already contains the continuous reconstruction and causal-held trailing
    dimensions. Only the first trailing dimension is changed. A nonnegative offset
    writes its paired block-ending command from that offset onward. Any incomplete
    trailing block is left untouched.
    """
    chunk = np.asarray(chunk)
    if chunk.ndim != 2:
        raise ValueError(f"chunk must have shape (T, A), got {chunk.shape}")
    if not isinstance(transition_summary, (tuple, list)) or len(transition_summary) != 2:
        raise ValueError("transition_summary must be an (offsets, end_values) pair")
    transition_offsets = np.asarray(transition_summary[0])
    transition_end_values = np.asarray(transition_summary[1])
    if transition_offsets.ndim != 1:
        raise ValueError(
            f"transition_offsets must have shape (B,), got {transition_offsets.shape}"
        )
    if transition_end_values.ndim != 1:
        raise ValueError(
            f"transition_end_values must have shape (B,), got {transition_end_values.shape}"
        )
    if not 0 <= n_hold <= chunk.shape[1]:
        raise ValueError(f"n_hold must be in [0, {chunk.shape[1]}], got {n_hold}")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    if not np.issubdtype(transition_offsets.dtype, np.integer):
        raise ValueError("transition_offsets must be an integer array")

    out = chunk.copy()
    if n_hold == 0 or len(out) == 0:
        return out.astype(np.float32, copy=False)

    n_blocks = len(out) // k
    if len(transition_offsets) != n_blocks:
        raise ValueError(
            "transition_offsets must contain one entry per complete block: "
            f"expected {n_blocks}, got {len(transition_offsets)}"
        )
    if len(transition_end_values) != n_blocks:
        raise ValueError(
            "transition_end_values must contain one entry per complete block: "
            f"expected {n_blocks}, got {len(transition_end_values)}"
        )
    if np.any((transition_offsets < -1) | (transition_offsets >= k)):
        raise ValueError(f"transition offsets must be in [-1, {k - 1}]")

    gripper_dim = chunk.shape[1] - n_hold
    for block_index, (offset, end_value) in enumerate(
        zip(transition_offsets, transition_end_values)
    ):
        if offset < 0:
            continue
        block_start = block_index * k
        block_stop = block_start + k
        out[block_start + offset : block_stop, gripper_dim] = end_value
    return out.astype(np.float32, copy=False)

RESAMPLERS = {
    "zoh": lambda block_sum, k: resample_zoh(block_sum, k),
    "tac_fold": resample_tac_fold,
    "tac_fold_satfix": resample_tac_fold_satfix,
    "spline": resample_spline,
    "spline_satfix": resample_spline_satfix,
    "pchip": resample_pchip,
    "pchip_satfix": resample_pchip_satfix,
    "bspline": resample_bspline,
    "bspline_satfix": resample_bspline_satfix,
    # Registry entries have the continuous (block_sum, k) interface.  The full
    # gripper_sync composition is applied by harness.apply_arm after this default
    # spline+satfix backbone has reconstructed the continuous dimensions.
    "gripper_sync": resample_spline_satfix,
    # "arc" (resample_arc, "Adaptive Rate-optimal Conservative fold") is RETIRED. ARC == qp_anchor.
    # It is no longer a selectable arm. Use --arms qp_anchor. (Old result JSONs with an "arc" arm are
    # historical records of the retired resampler, not the paper's ARC.)
}


def decimate_and_resample(delta_steps: np.ndarray, k: int, resampler: str) -> np.ndarray:
    """Full k-step chunk (T,D) in -> reconstructed (n_blocks*k, D) out.

    Any trailing remainder (len % k != 0) is left untouched at native rate, appended as-is.
    """
    n_blocks = len(delta_steps) // k
    remainder = delta_steps[n_blocks * k :]
    if n_blocks == 0:
        return delta_steps.copy()
    block_sum = coarsen_delta(delta_steps, k)
    fn = RESAMPLERS[resampler]
    reconstructed = fn(block_sum, k)
    if len(remainder) > 0:
        return np.concatenate([reconstructed, remainder], axis=0)
    return reconstructed
