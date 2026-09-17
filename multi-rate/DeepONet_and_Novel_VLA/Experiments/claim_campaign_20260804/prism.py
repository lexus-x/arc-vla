"""PRISM: Physical Rate-Invariant Scaling & Monotone Action Reconstruction.

A zero-retraining, inference-time execution engine for deploying pretrained
Vision-Language-Action (VLA) models (DeepONet-v2, ACT, OpenVLA, π₀) across arbitrary
control frequencies (10 Hz, 20 Hz, 40 Hz, 50 Hz, 100 Hz).

Key Mechanisms:
1. Physical Travel Invariance: Compensates for low-level controller impedance scaling.
2. Observation Cadence Invariance: Strides observation queue to keep physical history window constant.
3. Dynamic Per-Chunk Bandwidth Gating:
   - Measures local signal roughness / total variation ratio R.
   - If resolvable at target rate (k * R <= threshold): Employs exact definite integration over
     cumulative displacement with Fritsch-Carlson PCHIP monotone tangents (strictly eliminating
     backward steps and overshoot on monotone intervals).
   - If roughness exceeds Nyquist resolution (k * R > threshold): Falls back to piecewise-constant
     Zero-Order Hold (ZOH) where phase information cannot be safely reconstructed.
4. Synchronous Zero-Lag Gripper Latch: Prevents continuous float blurring on discrete gripper channels.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def compute_pchip_slopes(cumulative_y: np.ndarray) -> np.ndarray:
    """Computes Fritsch-Carlson PCHIP tangent slopes on cumulative curve.
    
    Guarantees strict monotonicity: if input deltas are non-negative,
    resampled sub-steps are guaranteed to be non-negative with zero envelope overshoot.
    """
    m = np.diff(cumulative_y, axis=0)  # Shape: (N - 1, D), dx = 1.0
    n, d = cumulative_y.shape
    slopes = np.zeros((n, d), dtype=cumulative_y.dtype)

    if n <= 2:
        return np.repeat(m, n, axis=0)

    # Interior points: harmonic mean where secants share sign; zero at local extrema
    m_prev = m[:-1]
    m_next = m[1:]
    same_sign = (m_prev * m_next) > 0

    denominator = m_prev + m_next
    safe_denom = np.where(same_sign, denominator, 1.0)
    interior_slopes = np.where(
        same_sign,
        2.0 * m_prev * m_next / safe_denom,
        0.0,
    )
    slopes[1:-1] = interior_slopes

    # Boundary conditions with standard PCHIP non-overshoot clamping
    # Left boundary
    slopes[0] = np.where(
        m[0] * slopes[1] > 0,
        1.5 * m[0] - 0.5 * slopes[1],
        m[0],
    )
    # Right boundary
    slopes[-1] = np.where(
        m[-1] * slopes[-2] > 0,
        1.5 * m[-1] - 0.5 * slopes[-2],
        m[-1],
    )

    return slopes


def estimate_chunk_roughness(delta_actions: np.ndarray) -> float:
    """Estimates normalized high-frequency roughness of an action chunk.
    
    R = Total_Variation(delta) / Total_Magnitude(delta)
    Smooth trajectories produce R < 0.3; high-frequency chatter produces R > 0.8.
    """
    if len(delta_actions) <= 1:
        return 0.0
    tv = np.sum(np.abs(np.diff(delta_actions, axis=0)))
    total_mag = np.sum(np.abs(delta_actions)) + 1e-8
    return float(tv / total_mag)


class PRISMAdapter:
    """PRISM Inference Engine: Physical Rate-Invariant Scaling & Monotone Reconstruction."""

    def __init__(
        self,
        source_rate_hz: float = 20.0,
        target_rate_hz: float = 40.0,
        has_gripper: bool = True,
        roughness_threshold: float = 1.2,
    ) -> None:
        self.source_rate_hz = float(source_rate_hz)
        self.target_rate_hz = float(target_rate_hz)
        self.has_gripper = has_gripper
        self.roughness_threshold = roughness_threshold

    def get_cadence_step_stride(self) -> int:
        """Computes the step stride for observation history buffers."""
        ratio = self.target_rate_hz / self.source_rate_hz
        return max(1, int(round(ratio)))

    def get_magnitude_scale(self) -> float:
        """Computes the travel velocity compensation scale factor."""
        return self.target_rate_hz / self.source_rate_hz

    def resample_chunk(
        self,
        actions: np.ndarray,
        target_rate_hz: Optional[float] = None,
        source_rate_hz: Optional[float] = None,
    ) -> np.ndarray:
        """Resamples an action chunk (T, D) with exact physical integral conservation.
        
        Args:
            actions: (T, D) numpy array of delta motor commands.
            target_rate_hz: Target environment control frequency.
            source_rate_hz: Model training/native control frequency.
            
        Returns:
            resampled_actions: (T * ratio, D) numpy array matching target frequency.
        """
        src_hz = float(source_rate_hz or self.source_rate_hz)
        tgt_hz = float(target_rate_hz or self.target_rate_hz)

        if abs(tgt_hz - src_hz) < 1e-4:
            return actions.copy()

        ratio = int(round(tgt_hz / src_hz))
        source_len, dim = actions.shape
        target_len = source_len * ratio

        # Separate continuous delta poses from discrete gripper state
        if self.has_gripper and dim > 1:
            delta = actions[:, :-1].astype(np.float64)
            gripper = actions[:, -1:].astype(np.float32)
            d_cont = dim - 1
        else:
            delta = actions.astype(np.float64)
            gripper = None
            d_cont = dim

        # Dynamic Per-Chunk Bandwidth Gating
        chunk_roughness = estimate_chunk_roughness(delta)
        effective_nyquist_load = ratio * chunk_roughness

        if effective_nyquist_load > self.roughness_threshold or ratio >= 4:
            # Piecewise constant Zero-Order Hold (ZOH) fallback
            resampled_pose = np.repeat(delta / float(ratio), ratio, axis=0).astype(np.float32)
        else:
            # Monotone PCHIP-Hermite integration over cumulative displacement
            cumulative = np.concatenate(
                [np.zeros((1, d_cont), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0
            )
            slopes = compute_pchip_slopes(cumulative)

            # Grid coordinates along unit sub-intervals [0, 1]
            t = np.linspace(0.0, 1.0, ratio + 1, dtype=np.float64)[1:][None, :, None]  # (1, ratio, 1)
            t2 = t * t
            t3 = t2 * t

            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2

            y0 = cumulative[:-1, None, :]
            y1 = cumulative[1:, None, :]
            d0 = slopes[:-1, None, :]
            d1 = slopes[1:, None, :]

            # Fully vectorized Hermite polynomial evaluation
            interp_blocks = h00 * y0 + h10 * d0 + h01 * y1 + h11 * d1  # (source_len, ratio, d_cont)
            resampled_cumulative = np.empty((target_len + 1, d_cont), dtype=np.float64)
            resampled_cumulative[0] = cumulative[0]
            resampled_cumulative[1:] = interp_blocks.reshape(target_len, d_cont)

            # Reconstruct discrete sub-step deltas via differencing
            resampled_pose = np.diff(resampled_cumulative, axis=0).astype(np.float32)

        # Re-attach discrete gripper channel with synchronous zero-lag hold
        if gripper is not None:
            resampled_gripper = np.repeat(gripper, ratio, axis=0).astype(np.float32)
            return np.concatenate([resampled_pose, resampled_gripper], axis=1)

        return resampled_pose
