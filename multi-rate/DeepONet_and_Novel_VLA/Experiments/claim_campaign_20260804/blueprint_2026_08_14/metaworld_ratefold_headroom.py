"""Measure whether MetaWorld 80 Hz actions leave multi-rate decoder headroom.

This is an evidence gate, not a policy evaluation. It reads only action columns
from the locally cached LeRobot parquet shards and evaluates fixed 0.4-second
windows. Run numerical analysis on Blackwell only.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator
from scipy.signal import periodogram


FPS = 80
WINDOW_STEPS = 32
RATES = (5, 10, 20, 30, 40)


def interval_averages(signal: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Exact averages of a unit-bin piecewise-constant signal."""
    rows = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        total = np.zeros(signal.shape[1], dtype=np.float64)
        first = int(np.floor(left))
        last = int(np.ceil(right))
        for index in range(first, last):
            overlap = max(0.0, min(right, index + 1.0) - max(left, index))
            if overlap:
                total += overlap * signal[min(index, len(signal) - 1)]
        rows.append(total / (right - left))
    return np.stack(rows)


def project_box_sum(values: np.ndarray, target_sum: np.ndarray) -> np.ndarray:
    """Euclidean projection per channel onto [-1,1] with a fixed sum."""
    output = np.empty_like(values)
    for dim in range(values.shape[1]):
        target = float(target_sum[dim])
        if not (-len(values) - 1e-8 <= target <= len(values) + 1e-8):
            raise ValueError("infeasible box-sum target")
        low = float(np.min(-1.0 - values[:, dim]) - 1.0)
        high = float(np.max(1.0 - values[:, dim]) + 1.0)
        for _ in range(64):
            midpoint = 0.5 * (low + high)
            current = np.clip(values[:, dim] + midpoint, -1.0, 1.0).sum()
            if current < target:
                low = midpoint
            else:
                high = midpoint
        output[:, dim] = np.clip(values[:, dim] + 0.5 * (low + high), -1.0, 1.0)
    return output


def decode_window(fine: np.ndarray, rate: int) -> dict[str, float]:
    target_steps = int(round(0.4 * rate))
    boundaries = np.linspace(0.0, WINDOW_STEPS, target_steps + 1)
    coarse = interval_averages(fine, boundaries)

    centers = np.arange(WINDOW_STEPS, dtype=np.float64) + 0.5
    cell_index = np.searchsorted(boundaries[1:], centers, side="right")
    cell_index = np.minimum(cell_index, target_steps - 1)
    zoh = coarse[cell_index]

    widths = np.diff(boundaries)
    cumulative = np.concatenate(
        [np.zeros((1, fine.shape[1])), np.cumsum(coarse * widths[:, None], axis=0)],
        axis=0,
    )
    fine_boundaries = np.arange(WINDOW_STEPS + 1, dtype=np.float64)
    pchip_path = PchipInterpolator(boundaries, cumulative, axis=0)(fine_boundaries)
    pchip_raw = np.diff(pchip_path, axis=0)
    integer_boundaries = bool(np.allclose(boundaries, np.round(boundaries)))
    if integer_boundaries:
        pchip = np.full_like(pchip_raw, np.nan)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            start = int(round(left))
            stop = int(round(right))
            target_sum = fine[start:stop].sum(axis=0)
            pchip[start:stop] = project_box_sum(pchip_raw[start:stop], target_sum)
        if not np.isfinite(pchip).all():
            raise AssertionError("integer-boundary PCHIP projection left unassigned bins")
    else:
        # The 30 Hz screen is reconstruction-only. A publication evaluator must
        # project on exact fractional intervals rather than discrete 80 Hz bins.
        pchip = np.clip(pchip_raw, -1.0, 1.0)

    return {
        "zoh_mse": float(np.mean((zoh - fine) ** 2)),
        "pchip_mse": float(np.mean((pchip - fine) ** 2)),
        "zoh_mae": float(np.mean(np.abs(zoh - fine))),
        "pchip_mae": float(np.mean(np.abs(pchip - fine))),
        "pchip_preprojection_clip_fraction": float(np.mean(np.abs(pchip_raw) > 1.0)),
        "exact_box_sum_projection": float(integer_boundaries),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
    }


def main() -> None:
    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    episodes: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    tasks: dict[int, int] = {}

    files = sorted(root.glob("*.parquet"))
    for path in files:
        table = pq.read_table(path, columns=["action", "episode_index", "frame_index", "task_index"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        for action, episode, frame, task in zip(
            actions,
            np.asarray(table["episode_index"]),
            np.asarray(table["frame_index"]),
            np.asarray(table["task_index"]),
        ):
            episodes[int(episode)].append((int(frame), action))
            tasks[int(episode)] = int(task)

    spectra: list[float] = []
    energy_above = {2.5: [], 5.0: [], 10.0: [], 15.0: [], 20.0: []}
    metrics = {rate: defaultdict(list) for rate in RATES}
    window_count = 0

    for episode, rows in episodes.items():
        rows.sort(key=lambda item: item[0])
        actions = np.clip(np.stack([item[1] for item in rows]), -1.0, 1.0)
        continuous = actions[:, :3]
        for dim in range(continuous.shape[1]):
            frequencies, power = periodogram(continuous[:, dim], fs=FPS, detrend="constant")
            total = float(power[1:].sum())
            if total <= 1e-12:
                continue
            cumulative = np.cumsum(power[1:]) / total
            spectra.append(float(frequencies[1:][np.searchsorted(cumulative, 0.95)]))
            for threshold in energy_above:
                energy_above[threshold].append(float(power[frequencies > threshold].sum() / total))

        for start in range(0, len(continuous) - WINDOW_STEPS + 1, WINDOW_STEPS):
            fine = continuous[start : start + WINDOW_STEPS]
            window_count += 1
            for rate in RATES:
                result = decode_window(fine, rate)
                for key, value in result.items():
                    metrics[rate][key].append(value)

    payload = {
        "protocol": {
            "fps": FPS,
            "window_steps": WINDOW_STEPS,
            "window_seconds": WINDOW_STEPS / FPS,
            "source_action_processing": "clip [-1,1], continuous dims 0:3 only",
            "files": len(files),
            "episodes": len(episodes),
            "tasks": len(set(tasks.values())),
            "windows": window_count,
        },
        "spectrum": {
            "f95_hz": quantiles(spectra),
            "energy_above_hz": {str(key): quantiles(value) for key, value in energy_above.items()},
        },
        "decoder_metrics": {
            str(rate): {key: quantiles(value) for key, value in rate_metrics.items()}
            for rate, rate_metrics in metrics.items()
        },
    }
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
