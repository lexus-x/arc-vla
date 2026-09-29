#!/usr/bin/env python3
"""Jev-supervised adaptive-rate wrapper for an existing VLA action policy."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Protocol
import numpy as np

from resample_math import decimate_and_resample
from pathlib import Path


def _load_env_key():
    if "TYPESAFE_API_KEY" not in os.environ:
        for env_path in [Path(__file__).resolve().parent / ".env", Path.cwd() / ".env", Path.home() / "Desktop" / ".env"]:
            if env_path.exists():
                try:
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("TYPESAFE_API_KEY="):
                                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                                break
                except Exception:
                    pass


_load_env_key()


PHASES = ("approach", "align", "grasp", "lift", "transport", "place", "recover")
RESOLUTIONS = {"k1": 1, "k2": 2, "k4": 4}
DECODERS = {"native", "zoh", "tac_fold_satfix"}


@dataclass(frozen=True)
class Decision:
    observation_id: int
    phase: str
    k: int
    decoder: str
    unsafe_probability: float
    goal_complete_probability: float
    ambiguity: float
    stop: bool = False
    fallback_reason: str | None = None


class Supervisor(Protocol):
    def decide(self, state: dict[str, Any]) -> Decision: ...


def questions() -> dict[str, Any]:
    """Build independent Jev judgments evaluated together over one state."""
    from typesafe_sdk import Choice, Noul, Score

    return {
        "phase": Choice(
            instructions="Which manipulation phase best matches the current observation?",
            criteria={phase: None for phase in PHASES},
        ),
        "resolution": Choice(
            instructions="Which action resolution is safest while retaining useful temporal coverage?",
            criteria={
                "k1": "Native resolution for precision, uncertainty, contact, or recovery.",
                "k2": "Moderate compression for a clear approach or ordinary motion.",
                "k4": "Deep compression only for clear, low-risk free-space motion.",
            },
        ),
        "decoder": Choice(
            instructions="Which approved execution decoder best fits this state?",
            criteria={
                "native": "No temporal reconstruction; required for k1.",
                "zoh": "Equal split when piecewise-constant execution is appropriate.",
                "tac_fold_satfix": "Conservative smooth reconstruction with bounded saturation repair.",
            },
        ),
        "unsafe": Noul(
            instructions="Would continuing autonomous motion from this state create meaningful collision, loss-of-control, or task-damage risk?"
        ),
        "goal_complete": Noul(
            instructions="Does the supplied state indicate that the requested manipulation goal is complete?"
        ),
        "ambiguity": Score(
            instructions="How ambiguous is the scene and manipulation state?",
            criteria=[
                "Clear target, relation, and progress.",
                "Minor uncertainty unlikely to change the action.",
                "Material uncertainty about target, phase, or geometry.",
                "Insufficient or contradictory state; autonomous action should not proceed.",
            ],
        ),
    }


def conservative_fallback(observation_id: int, reason: str) -> Decision:
    return Decision(observation_id, "recover", 1, "native", 0.0, 0.0, 3.0, False, reason)


class JevSupervisor:
    """Live Jev semantic supervisor; the VLA remains the action generator."""

    def __init__(self, *, model: str = "jev-1.13.0", min_confidence: float = 0.65) -> None:
        self.model = model
        self.min_confidence = min_confidence

    def decide(self, state: dict[str, Any]) -> Decision:
        from typesafe_sdk import TypeSafeClient

        observation_id = int(state["observation_id"])
        try:
            with TypeSafeClient() as client:
                response = client.system_one(state=state, questions=questions(), model=self.model)
        except Exception as error:
            return conservative_fallback(observation_id, f"jev_error:{type(error).__name__}")

        answers = response.answers
        phase, resolution, decoder = answers["phase"], answers["resolution"], answers["decoder"]
        low_confidence = min(phase.confidence, resolution.confidence, decoder.confidence) < self.min_confidence
        unsafe = float(answers["unsafe"].noul)
        complete = float(answers["goal_complete"].noul)
        ambiguity = float(answers["ambiguity"].score)
        stop = unsafe >= 0.7 or complete >= 0.8 or ambiguity >= 2.5
        if stop:
            return Decision(observation_id, phase.choice, 1, "native", unsafe, complete, ambiguity, True)
        if low_confidence or ambiguity >= 2.0:
            return Decision(
                observation_id,
                phase.choice,
                1,
                "native",
                unsafe,
                complete,
                ambiguity,
                False,
                "low_confidence_or_ambiguity",
            )
        k = RESOLUTIONS.get(resolution.choice, 1)
        selected_decoder = decoder.choice if decoder.choice in DECODERS else "native"
        if k == 1:
            selected_decoder = "native"
        elif selected_decoder == "native":
            selected_decoder = "tac_fold_satfix"
        return Decision(observation_id, phase.choice, k, selected_decoder, unsafe, complete, ambiguity)


class DemoSupervisor:
    """Deterministic local stand-in that exercises the same execution contract."""

    def decide(self, state: dict[str, Any]) -> Decision:
        observation_id = int(state["observation_id"])
        phase = str(state.get("phase", "recover"))
        unsafe = float(bool(state.get("unsafe")))
        complete = float(bool(state.get("goal_complete")))
        ambiguity = float(state.get("ambiguity", 0.0))
        if unsafe or complete or ambiguity >= 2.5:
            return Decision(observation_id, phase, 1, "native", unsafe, complete, ambiguity, True)
        if ambiguity >= 2.0 or phase in {"align", "grasp", "place", "recover"}:
            return Decision(observation_id, phase, 1, "native", unsafe, complete, ambiguity)
        k = 4 if phase == "transport" else 2
        return Decision(observation_id, phase, k, "tac_fold_satfix", unsafe, complete, ambiguity)


class AsyncSupervisor:
    """One in-flight request, latest observation queued; the control loop never blocks."""

    def __init__(self, supervisor: Supervisor) -> None:
        self.supervisor = supervisor
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-supervisor")
        self.future: Future[Decision] | None = None
        self.pending: dict[str, Any] | None = None
        self.started_at = 0.0
        self.latencies_ms: list[float] = []

    def _start(self, state: dict[str, Any]) -> None:
        self.started_at = time.perf_counter()
        self.future = self.executor.submit(self.supervisor.decide, state)

    def submit(self, state: dict[str, Any]) -> None:
        snapshot = dict(state)
        if self.future is None:
            self._start(snapshot)
        else:
            self.pending = snapshot

    def poll(self, current_observation_id: int) -> Decision | None:
        if self.future is None or not self.future.done():
            return None
        decision = self.future.result()
        self.latencies_ms.append(1000 * (time.perf_counter() - self.started_at))
        self.future = None
        if self.pending is not None:
            newest, self.pending = self.pending, None
            self._start(newest)
        return decision if decision.observation_id == current_observation_id else None

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "AsyncSupervisor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def apply_decision(
    action_chunk: np.ndarray,
    decision: Decision,
    *,
    current_observation_id: int,
    continuous_dims: int,
) -> np.ndarray:
    """Apply a fresh decision; stale or unsafe decisions produce conservative commands."""
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or not 0 < continuous_dims <= chunk.shape[1]:
        raise ValueError("action_chunk must be (time, action), with valid continuous_dims")
    if decision.observation_id != current_observation_id:
        decision = conservative_fallback(current_observation_id, "stale_response")
    if decision.stop:
        stopped = np.zeros_like(chunk)
        if continuous_dims < chunk.shape[1]:
            stopped[:, continuous_dims:] = chunk[0, continuous_dims:]
        return stopped
    if decision.k == 1 or decision.decoder == "native":
        return chunk.copy()
    if decision.k not in RESOLUTIONS.values() or decision.decoder not in DECODERS:
        raise ValueError(f"unapproved decision: k={decision.k}, decoder={decision.decoder}")
    continuous = decimate_and_resample(chunk[:, :continuous_dims], decision.k, decision.decoder)
    if continuous_dims == chunk.shape[1]:
        return continuous
    blocks = len(chunk) // decision.k
    held = np.repeat(chunk[: blocks * decision.k : decision.k, continuous_dims:], decision.k, axis=0)
    if blocks * decision.k < len(chunk):
        held = np.concatenate((held, chunk[blocks * decision.k :, continuous_dims:]), axis=0)
    return np.concatenate((continuous, held), axis=1).astype(np.float32)


def demo(supervisor: Supervisor) -> list[dict[str, Any]]:
    chunk = np.array(
        [[0.15 + 0.02 * step, -0.08 + 0.01 * step, 1.0] for step in range(8)],
        dtype=np.float32,
    )
    states: list[dict[str, Any]] = [
        {"observation_id": 1, "instruction": "pick the cube", "phase": "approach", "ambiguity": 0},
        {"observation_id": 2, "instruction": "carry the cube", "phase": "transport", "ambiguity": 0},
        {"observation_id": 3, "instruction": "place the cube", "phase": "place", "ambiguity": 3},
    ]
    rows = []
    for state in states:
        decision = supervisor.decide(state)
        output = apply_decision(
            chunk,
            decision,
            current_observation_id=int(state["observation_id"]),
            continuous_dims=2,
        )
        rows.append(
            {
                "decision": asdict(decision),
                "first_action": output[0].round(4).tolist(),
                "continuous_sum_error": None
                if decision.stop
                else float(np.abs(output[:, :2].sum(0) - chunk[:, :2].sum(0)).max()),
            }
        )
    return rows


def async_demo(supervisor: Supervisor) -> dict[str, Any]:
    """Submit two snapshots immediately and show that only the newest result is accepted."""
    with AsyncSupervisor(supervisor) as async_supervisor:
        async_supervisor.submit({"observation_id": 10, "phase": "approach", "ambiguity": 0})
        async_supervisor.submit({"observation_id": 11, "phase": "transport", "ambiguity": 0})
        decision = None
        deadline = time.monotonic() + 30
        while decision is None and time.monotonic() < deadline:
            decision = async_supervisor.poll(11)
            if decision is None:
                time.sleep(0.005)
        if decision is None:
            raise TimeoutError("supervisor did not return the newest observation within 30 seconds")
        return {"accepted": asdict(decision), "latencies_ms": async_supervisor.latencies_ms}


def self_test() -> None:
    rows = demo(DemoSupervisor())
    assert [row["decision"]["k"] for row in rows] == [2, 4, 1]
    assert rows[2]["decision"]["stop"] is True
    assert max(row["continuous_sum_error"] for row in rows if not row["decision"]["stop"]) < 1e-6
    chunk = np.ones((4, 2), np.float32)
    stale = Decision(1, "transport", 4, "tac_fold_satfix", 0, 0, 0)
    assert np.array_equal(apply_decision(chunk, stale, current_observation_id=2, continuous_dims=2), chunk)
    async_result = async_demo(DemoSupervisor())
    assert async_result["accepted"]["observation_id"] == 11
    assert len(async_result["latencies_ms"]) == 2
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call Jev; requires typesafe-sdk and TYPESAFE_API_KEY")
    parser.add_argument("--async-demo", action="store_true", help="exercise latest-only non-blocking supervision")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.live and not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is required for --live")
    supervisor: Supervisor = JevSupervisor() if args.live else DemoSupervisor()
    print(json.dumps(async_demo(supervisor) if args.async_demo else demo(supervisor), indent=2))


if __name__ == "__main__":
    main()
