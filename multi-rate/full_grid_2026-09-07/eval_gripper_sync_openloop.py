"""Open-loop RoboCasa gripper reconstruction check from cached demonstrations.

The raw gripper trace is used as the scoring target, but the synchronizer receives only
the per-block offset/value summary produced by ``coarsen_gripper_transitions``.

Run from this directory:
    python eval_gripper_sync_openloop.py --k 4

The fold caches overlap. Episodes are deduplicated by their action-array content so
each demonstrated trajectory contributes once to its task-level result.
"""
import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

from resample_math import coarsen_gripper_transitions, gripper_sync


BASELINES = ("zoh", "spline_satfix", "bspline_eps_satfix")
TASK_FROM_CACHE = re.compile(r"^demos_(RC-.+?)_\d+_f\d+\.npz$")


def causal_hold(raw_chunk: np.ndarray, n_hold: int, k: int) -> np.ndarray:
    """Reproduce apply_arm's trailing-dimension hold without touching continuous dims."""
    out = raw_chunk.copy()
    if n_hold == 0:
        return out
    nd = out.shape[1] - n_hold
    for start in range(0, len(out), k):
        stop = min(start + k, len(out))
        out[start:stop, nd:] = raw_chunk[start, nd:]
    return out


def action_windows(actions: np.ndarray, horizon: int = 8):
    """Yield the same edge-padded raw execution chunks used by the policy harness."""
    padded = np.concatenate([actions, np.repeat(actions[-1:], horizon - 1, axis=0)], axis=0)
    for start in range(len(actions)):
        yield padded[start : start + horizon]


def unique_cached_actions(paths):
    tasks = {}
    seen = {}
    for path in sorted(paths):
        match = TASK_FROM_CACHE.match(os.path.basename(path))
        if not match:
            continue
        task = match.group(1)
        tasks.setdefault(task, [])
        seen.setdefault(task, set())
        with np.load(path, allow_pickle=True) as cache:
            for actions in cache["A"]:
                actions = np.asarray(actions, dtype=np.float32)
                digest = hashlib.sha256(actions.shape.__repr__().encode() + actions.tobytes()).digest()
                if digest not in seen[task]:
                    seen[task].add(digest)
                    tasks[task].append(actions)
    return tasks


def evaluate_task(episodes, k: int, n_hold: int = 6):
    matches = {name: 0 for name in (*BASELINES, "gripper_sync")}
    total = 0
    changed_windows = 0
    gripper_dim = episodes[0].shape[1] - n_hold
    for actions in episodes:
        for raw in action_windows(actions):
            held = causal_hold(raw, n_hold, k)
            target = raw[:, gripper_dim]
            transition_summary = coarsen_gripper_transitions(target, k)
            synced = gripper_sync(held, transition_summary, n_hold, k)
            baseline_match = int(np.count_nonzero(held[:, gripper_dim] == target))
            for name in BASELINES:
                matches[name] += baseline_match
            matches["gripper_sync"] += int(np.count_nonzero(synced[:, gripper_dim] == target))
            total += len(target)
            changed_windows += int(baseline_match != len(target))
    return {
        "episodes": len(episodes),
        "windows": total // 8,
        "windows_changed_by_hold": changed_windows,
        "gripper_steps": total,
        "accuracy": {name: matches[name] / total for name in matches},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--cache-glob", default="demos_RC-*_39_f*.npz")
    parser.add_argument("--output", default="result_gripper_sync_openloop.json")
    args = parser.parse_args()

    tasks = unique_cached_actions(glob.glob(args.cache_glob))
    if not tasks:
        raise SystemExit(f"no caches matched {args.cache_glob!r}")
    results = {task: evaluate_task(episodes, args.k) for task, episodes in sorted(tasks.items())}
    payload = {"k": args.k, "horizon": 8, "n_hold": 6, "tasks": results}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    methods = (*BASELINES, "gripper_sync")
    print("task".ljust(30) + "".join(f"{name:>24}" for name in methods))
    for task, result in results.items():
        print(task.ljust(30) + "".join(f"{100 * result['accuracy'][name]:23.3f}%" for name in methods))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
