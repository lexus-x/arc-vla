"""Timing/equivalence probe only; never produces benchmark success-rate claims.

Run with the ms3 Python. Keeps the existing checkpoint, precision, DDIM steps,
resamplers, reset logic and episode RNG; tests CUDA graph replay before rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import harness as h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("episodes must be positive")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    z = np.load(root / f"demos_{args.task}_200.npz", allow_pickle=True)
    obs, actions = np.concatenate(z["O"]), np.concatenate(z["A"])
    onorm, anorm = h.MinMax(obs), h.MinMax(actions)
    checkpoint = root / f"dp_{args.task}.pt"
    policy = h.DiffusionPolicy(obs.shape[-1], actions.shape[-1])
    policy.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    policy.cuda().eval()
    eager = policy.sample
    static_obs = torch.as_tensor(onorm.norm(obs[:2])[None], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            eager(static_obs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = eager(static_obs)

    def replay(o):
        static_obs.copy_(o)
        graph.replay()
        return static_output

    report = {"kind": "throughput_probe_not_SR", "task": args.task,
              "checkpoint_sha256": h.sha256_file(checkpoint), "episodes": args.episodes,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "inference_checks": [], "rollouts": {}}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for i in range(8):
        o = torch.as_tensor(onorm.norm(obs[i * 2:i * 2 + 2])[None], device="cuda")
        seed = 1_000_003 * i + 3
        torch.manual_seed(seed)
        ref = eager(o).clone()
        torch.manual_seed(seed)
        got = replay(o).clone()
        report["inference_checks"].append({"seed": seed, "equal": torch.equal(ref, got),
                                           "max_abs_error": (ref - got).abs().max().item()})
    save()
    print("inference equivalence:", report["inference_checks"], flush=True)
    if not all(x["equal"] for x in report["inference_checks"]):
        raise RuntimeError("Graph fails exact-output gate; no accelerated rollouts permitted")

    for name, sample in (("eager", eager), ("graph", replay)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(20):
            torch.manual_seed(i)
            sample(static_obs)
        torch.cuda.synchronize()
        report[f"{name}_sample_seconds"] = (time.perf_counter() - start) / 20
    save()
    print("sample timings:", {k: v for k, v in report.items() if k.endswith("seconds")}, flush=True)

    h.K, h.ARMS = 4, ["qp", "tac_fold", "spline", "bspline_eps_raw"]
    # Development episodes only: confirmation starts at offset 100 (500 for PickCube).
    sim = h.ManiSkillSim(args.task)
    _, episodes = sim.split(200, args.episodes)
    original_step = sim.step
    trace = []

    def step(action):
        out = original_step(action)
        trace.append(np.asarray(action).tobytes() + out[0].tobytes() + bytes(out[1:]))
        return out

    sim.step = step
    try:
        for name, sample in (("eager", eager), ("graph", replay)):
            policy.sample = sample
            trace.clear()
            start = time.perf_counter()
            outcomes = [h.run_episode(sim, policy, onorm, anorm, "cuda", i, ep,
                                     "step", h.K, actions.shape[-1] - sim.n_hold, 4)
                        for i, ep in enumerate(episodes)]
            torch.cuda.synchronize()
            report["rollouts"][name] = {"seconds": time.perf_counter() - start,
                                        "steps": len(trace),
                                        "trace_sha256": hashlib.sha256(b"".join(trace)).hexdigest(),
                                        "outcomes": outcomes}
            save()
            print(name, report["rollouts"][name], flush=True)
    finally:
        sim.close()
    a, b = report["rollouts"]["eager"], report["rollouts"]["graph"]
    report["exact_rollout_match"] = a["trace_sha256"] == b["trace_sha256"]
    report["rollout_speedup"] = a["seconds"] / b["seconds"]
    save()
    assert report["exact_rollout_match"], "Actions/observations differ; gate failed"
    print("PASS: exact rollout trace match; speedup", report["rollout_speedup"], flush=True)


if __name__ == "__main__":
    main()
