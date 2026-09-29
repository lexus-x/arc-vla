# Jev-supervised adaptive-rate VLA

This prototype keeps the existing VLA as the action generator and uses Jev as a semantic
supervisor. Jev selects manipulation phase, action resolution, and an approved decoder while
independently judging unsafe state, goal completion, and ambiguity.

```text
vision/robot state -> structured JSON -> Jev -> deterministic gate
                                              -> existing VLA chunk -> existing resampler -> robot
```

Jev never emits actuator values. Stale responses are ignored, unsafe/completed/very ambiguous
states stop continuous motion, and low-confidence decisions fall back to native `k=1`. The
approved adaptive paths are `k=2` or `k=4` with `zoh` or `tac_fold_satfix`.

## Run locally

```bash
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
python3 jev_vla_supervisor.py --self-test
python3 jev_vla_supervisor.py
python3 jev_vla_supervisor.py --async-demo
```

The default demo is deterministic and makes no network calls. It shows approach (`k=2`),
transport (`k=4`), and an ambiguous placement that stops.

## Run with Jev

```bash
pip install typesafe-sdk
export TYPESAFE_API_KEY=...
python3 jev_vla_supervisor.py --live
```

The live example still uses synthetic action chunks and structured states. Before robot use,
connect `JevSupervisor.decide()` to a scene-state producer and call `apply_decision()` on the
existing policy chunk. `AsyncSupervisor` keeps one request in flight, retains only the newest
queued observation, and returns a decision only when its observation ID is current. The motor loop
therefore never waits on the network. Pinning `jev-1.13.0` makes calibration changes explicit.

## Required evaluation before deployment

Compare fixed `k`, a deterministic phase heuristic, Jev routing, and an oracle on paired seeds.
Record task success, p50/p95/p99 Jev latency, stale-response rate, fallback rate, decision flips,
unsafe stops, and API failures. Thresholds in this prototype are conservative starting values,
not validated safety guarantees.
