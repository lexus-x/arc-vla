"""Shared, explicit contract for chronological VLA state histories.

The history is always oldest -> current.  This module only constructs and
validates dataset requests; model/training code chooses how to consume it.
"""

from __future__ import annotations

from collections.abc import Mapping


STATE_KEY = "observation.state"
STATE_HISTORY_STEPS = 8
STATE_HISTORY_ORDER = "oldest_to_current"


def state_history_offsets(fps: float, steps: int = STATE_HISTORY_STEPS) -> list[float]:
    """Return the required oldest-to-current offsets in seconds."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if steps < 2:
        raise ValueError("state history needs at least two steps")
    return [frame / float(fps) for frame in range(1 - steps, 1)]


def request_state_history(
    delta_timestamps: Mapping[str, list[float]] | None,
    fps: float,
    *,
    steps: int = STATE_HISTORY_STEPS,
    state_key: str = STATE_KEY,
) -> dict[str, list[float]]:
    """Copy a LeRobot request and add the canonical state-history offsets."""
    request = dict(delta_timestamps or {})
    request[state_key] = state_history_offsets(fps, steps)
    validate_state_history_request(request, fps, steps=steps, state_key=state_key)
    return request


def validate_state_history_request(
    delta_timestamps: Mapping[str, list[float]] | None,
    fps: float,
    *,
    steps: int = STATE_HISTORY_STEPS,
    state_key: str = STATE_KEY,
) -> None:
    """Reject a request that would silently use another history convention."""
    if delta_timestamps is None or state_key not in delta_timestamps:
        raise ValueError(f"missing required {state_key!r} history request")
    actual = list(delta_timestamps[state_key])
    expected = state_history_offsets(fps, steps)
    if len(actual) != steps or any(abs(a - e) > 1e-9 for a, e in zip(actual, expected)):
        raise ValueError(
            f"{state_key} must be {steps} oldest-to-current frames {expected}; got {actual}"
        )


def validate_state_history(states, *, steps: int = STATE_HISTORY_STEPS) -> None:
    """Validate a collated batch shaped ``(batch, oldest_to_current, state_dim)``."""
    shape = getattr(states, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError("state history must have shape (batch, steps, state_dim)")
    if shape[0] < 1 or shape[1] != steps or shape[2] < 1:
        raise ValueError(f"state history must have nonempty shape (batch, {steps}, state_dim); got {tuple(shape)}")


def state_history_provenance(
    fps: float, *, steps: int = STATE_HISTORY_STEPS, state_key: str = STATE_KEY
) -> dict[str, object]:
    """JSON-safe provenance to store beside every history-enabled run."""
    return {
        "feature": state_key,
        "steps": steps,
        "order": STATE_HISTORY_ORDER,
        "offsets_frames": list(range(1 - steps, 1)),
        "offsets_seconds": state_history_offsets(fps, steps),
    }


if __name__ == "__main__":
    class _Batch:
        shape = (2, 8, 7)

    request = request_state_history({"action": [0.0]}, fps=20.0)
    validate_state_history_request(request, fps=20.0)
    validate_state_history(_Batch())
    assert state_history_provenance(20.0)["offsets_frames"] == [-7, -6, -5, -4, -3, -2, -1, 0]
    try:
        validate_state_history_request({STATE_KEY: [0.0]}, fps=20.0)
    except ValueError:
        pass
    else:
        raise AssertionError("single-frame state request was accepted")
    print("STATE_HISTORY_OK")
