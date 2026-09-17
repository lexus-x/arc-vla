"""RTC latency-robustness sweep on LIBERO-Spatial, for the flow and DeepONet heads.

Stressor: inference delay `d` in control steps at the benchmark's native 20 Hz. This is the
axis RTC (arXiv:2506.07339) occupies. It is NOT a control-rate sweep: LIBERO's control loop and
its teleoperated demonstrations are both 20 Hz, so there is no valid rate interval above native.

Arms
----
naive : chunk replaces the queue, first `d` entries discarded (hard splice). RTC's own degenerate
        baseline. Runs on both heads.
te    : ACT temporal ensembling (Zhao et al. 2023) over overlapping chunks, exp(-m*age) weights.
        The published soft-merge available to a single-pass head. Runs on both heads.
rtc   : official lerobot RTCProcessor guidance inside the flow denoising loop. FLOW ONLY --
        the DeepONet head's `sample_actions` is a single forward pass with no `x_t`/`v_t`
        trajectory to guide, so RTC has nothing to hook. Enforced by `_assert_arm_supported`.

Protocol is inherited from `evaluate_libero_standard.py` (Gate-0 protocol) with ONE recorded
deviation: execution horizon H replaces replan=1, because a delay sweep is undefined when only
one action is consumed per inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluate_libero_standard import (
    CONTROL_FREQ,
    DATASET,
    MAX_STEPS,
    N_TASKS,
    SUITE,
    _make_env,
    _optional_sha256,
    _parse_models,
    _policy_input,
    _sha256,
    _save_json,
    load_policy,
)

ARMS = ("naive", "te", "rtc")
DELAYS = (0, 1, 2, 3, 4)
EXECUTION_HORIZON = 10
TE_M = 0.01  # ACT temporal-ensembling decay; exp(-m * age), oldest prediction weighted highest
GUIDANCE_LIVE_TOL = 1e-6


def _assert_arm_supported(head: str, arm: str) -> None:
    """RTC requires an iterative denoiser. Fail closed rather than silently no-op."""
    if arm == "rtc" and head != "flow":
        raise SystemExit(
            f"[FATAL] arm 'rtc' requires a diffusion/flow denoising loop; head {head!r} is "
            "single-pass (sample_actions -> predict_chunk, no x_t/v_t trajectory). RTC would be "
            "an inert no-op. Use arm 'te' for the single-pass comparator."
        )
    if arm not in ARMS:
        raise SystemExit(f"[FATAL] unknown arm {arm!r}")


def _locked_rtc_protocol(horizon) -> dict:
    """Protocol identity for the manifest.

    Sample size (`--trials`) and the arm/delay subset are deliberately EXCLUDED: they are
    coverage, not protocol, and a later run must be able to extend a screening run rather than
    be rejected as a mixed resume. Everything that would invalidate a comparison is pinned.
    """
    return {
        "suite": SUITE,
        "n_tasks": N_TASKS,
        "max_steps": MAX_STEPS,
        "control_freq": CONTROL_FREQ,
        "execution_horizon": horizon,
        "te_decay_m": TE_M,
        "rollout_reset_seed": "1000 + trial_index",
        "deviation_from_standard": "execution_horizon replaces replan=1; a delay sweep is "
                                   "undefined when one action is consumed per inference",
        "videos": False,
    }


def _configure_rtc(policy, arm: str, horizon: int):
    """Attach the official RTCProcessor to a flow policy. No-op for other arms/heads."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    cfg = RTCConfig(enabled=(arm == "rtc"), execution_horizon=horizon)
    policy.config.rtc_config = cfg
    policy.init_rtc_processor()
    if hasattr(policy.model, "rtc_processor"):
        policy.model.rtc_processor = policy.rtc_processor
    if arm == "rtc" and policy.rtc_processor is None:
        raise SystemExit("[FATAL] arm 'rtc' requested but RTCProcessor failed to initialise")
    return cfg


def _predict_chunk(policy, batch, *, arm, delay, horizon, left_over):
    """One chunk from the policy, with RTC kwargs only on the rtc arm."""
    import torch

    if arm == "rtc":
        # NOT under no_grad: RTC guidance differentiates through the denoiser.
        return policy.predict_action_chunk(
            batch,
            prev_chunk_left_over=left_over,
            inference_delay=delay,
            execution_horizon=horizon,
        )
    with torch.no_grad():
        return policy.predict_action_chunk(batch)


def _assert_guidance_live(policy, batch, *, delay, horizon, left_over) -> float:
    """Prove RTC is doing something: guided chunk must differ from the unguided one.

    Ten inert-knob incidents are on record in this project. A knob that cannot be shown to
    change the output is indistinguishable from an absent knob.
    """
    import torch

    guided = _predict_chunk(policy, batch, arm="rtc", delay=delay, horizon=horizon,
                            left_over=left_over)
    with torch.no_grad():
        unguided = policy.predict_action_chunk(batch)
    diff = (guided.detach() - unguided).abs().max().item()
    if diff <= GUIDANCE_LIVE_TOL:
        raise SystemExit(
            f"[FATAL] RTC guidance is inert: max|guided - unguided| = {diff:.3e} at delay={delay}. "
            "Refusing to report a number from a no-op knob."
        )
    return diff


class _TemporalEnsemble:
    """ACT-style exponentially-weighted average over overlapping chunk predictions."""

    def __init__(self, m: float = TE_M):
        self.m = float(m)
        self.chunks: list[tuple[int, "object"]] = []  # (absolute start step, chunk tensor)

    def clear(self) -> None:
        self.chunks.clear()

    def add(self, start_step: int, chunk) -> None:
        self.chunks.append((start_step, chunk))

    def action(self, step: int):
        import torch

        picks, weights = [], []
        for age, (start, chunk) in enumerate(self.chunks):
            offset = step - start
            if 0 <= offset < chunk.shape[0]:
                picks.append(chunk[offset])
                weights.append(float(torch.exp(torch.tensor(-self.m * age))))
        if not picks:
            return None
        stacked = torch.stack(picks, dim=0)
        w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).reshape(-1, 1)
        self.chunks = [(s, c) for s, c in self.chunks if step - s < c.shape[0] - 1]
        return (stacked * w).sum(dim=0) / w.sum()


def _rollout_async(policy, preprocessor, postprocessor, env, task_description, seed, *,
                   arm, delay, horizon, guidance_probe):
    """One episode under an emulated inference delay of `delay` control steps.

    Timing model: inference for the next chunk is *started* `delay` steps before the current
    chunk's execution budget runs out, and the result is *installed* exactly when it does. The
    robot executes the previous chunk's actions throughout the delay window -- which is precisely
    the prefix RTC freezes.
    """
    import torch
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    if delay >= horizon:
        raise SystemExit(f"[FATAL] delay {delay} must be < execution horizon {horizon}")

    policy.reset()
    queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=horizon))
    ensemble = _TemporalEnsemble()
    observation, _ = env.reset(seed=seed)

    pending = None          # (chunk, request_step) awaiting installation
    pending_at = None       # absolute step at which it installs
    pending_index = 0       # queue index when inference started
    budget = 0              # actions left to consume from the installed chunk
    probe_value = None

    def batch_now():
        batch = preprocessor(_policy_input(observation, task_description))
        return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}

    for step in range(1, MAX_STEPS + 1):
        if pending is not None and step >= pending_at:
            if arm == "te":
                # Register at the REQUEST step: chunk[i] is the action for absolute time
                # request_step + i, so lookup at `step` naturally skips the `delay` entries
                # whose time has already elapsed -- the same skip `_replace_actions_queue`
                # performs via `clamped_delay` on the naive/rtc path.
                ensemble.add(pending[1], pending[0])
            else:
                queue.merge(pending[0], pending[0], delay, pending_index)
            pending, pending_at = None, None
            budget = horizon

        need_now = budget <= 0 and pending is None
        if need_now or (pending is None and budget <= delay):
            left_over = queue.get_left_over() if arm == "rtc" else None
            pending_index = queue.get_action_index()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = _predict_chunk(policy, batch_now(), arm=arm, delay=delay,
                                       horizon=horizon, left_over=left_over)
                if guidance_probe and arm == "rtc" and left_over is not None and probe_value is None:
                    probe_value = _assert_guidance_live(policy, batch_now(), delay=delay,
                                                        horizon=horizon, left_over=left_over)
            pending = (chunk.detach()[0], step)
            pending_at = step + (delay if not need_now else 0)
            if need_now:
                # Starvation path: nothing is executable, so this chunk installs synchronously
                # and its own request step is the current step -- offset 0, no skip.
                if arm == "te":
                    ensemble.add(pending[1], pending[0])
                else:
                    queue.merge(pending[0], pending[0], 0, pending_index)
                pending, pending_at = None, None
                budget = horizon

        action = ensemble.action(step) if arm == "te" else queue.get()
        if action is None:
            raise SystemExit(f"[FATAL] action starvation at step {step} (arm={arm}, delay={delay})")
        budget -= 1

        observation, _, terminated, truncated, info = env.step(
            postprocessor(action.unsqueeze(0)).to("cpu").float().numpy().reshape(-1))
        if info.get("is_success", False):
            return {"seed": seed, "success": True, "steps": step, "guidance_delta": probe_value}
        if terminated or truncated:
            break
    return {"seed": seed, "success": False, "steps": step, "guidance_delta": probe_value}


def _manifest(models, stats_path, horizon) -> dict:
    root = Path(__file__).resolve().parent
    files = ("evaluate_rtc_latency.py", "evaluate_libero_standard.py",
             "modeling_smolvla_deeponet_v2.py", "deeponet_head_v2.py")
    import lerobot.policies.rtc.modeling_rtc as modeling_rtc

    return {
        "protocol": _locked_rtc_protocol(horizon),
        "dataset": DATASET,
        "stats_path": str(Path(stats_path).resolve()) if stats_path else None,
        "stats_sha256": _optional_sha256(Path(stats_path)) if stats_path else None,
        "code_sha256": {name: _optional_sha256(root / name) for name in files},
        "lerobot_rtc_sha256": _optional_sha256(Path(modeling_rtc.__file__)),
        "models": {name: {"head": head, "checkpoint": str(Path(ckpt).resolve()),
                          "model_sha256": _sha256(Path(ckpt) / "model.safetensors")}
                   for name, (head, ckpt) in models.items()},
        "runtime": {k: os.environ.get(k) for k in sorted(os.environ)
                    if k.startswith("DEEPONET_") or k == "TEMPO_SPEED"},
    }


def _selfcheck() -> None:
    assert _locked_rtc_protocol(EXECUTION_HORIZON)["max_steps"] == 220
    assert _locked_rtc_protocol(EXECUTION_HORIZON)["control_freq"] == 20
    assert "trials_per_task" not in _locked_rtc_protocol(EXECUTION_HORIZON)
    for head in ("deeponet", "til", "asrc"):
        try:
            _assert_arm_supported(head, "rtc")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"rtc must be rejected for single-pass head {head}")
    _assert_arm_supported("flow", "rtc")
    for head in ("flow", "deeponet"):
        _assert_arm_supported(head, "naive")
        _assert_arm_supported(head, "te")
    ens = _TemporalEnsemble(m=0.0)
    assert ens.action(0) is None, "empty ensemble must yield no action"

    # TE absolute-time alignment. chunk[i] holds the action for time request_step + i, so a chunk
    # requested at step 5 and read at step 7 must return index 2 -- i.e. a delay of 2 skips two
    # entries, exactly as `_replace_actions_queue` does on the naive/rtc path. Registering at the
    # INSTALL step instead returns index 0 and silently replays stale actions.
    import torch

    ens = _TemporalEnsemble(m=0.0)
    ens.add(5, torch.arange(10, dtype=torch.float32).reshape(10, 1))
    got = float(ens.action(7).item())
    assert got == 2.0, f"TE misaligned: expected index 2 at step 7 for a chunk requested at 5, got {got}"
    print("SELF_CHECK_OK", len(ARMS), len(DELAYS), "te_align=OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="NAME=flow|deeponet=CHECKPOINT")
    parser.add_argument("--arm", action="append", choices=list(ARMS))
    parser.add_argument("--delays", default=",".join(str(d) for d in DELAYS))
    parser.add_argument("--horizon", type=int, default=EXECUTION_HORIZON)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out")
    parser.add_argument("--stats_path")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if not args.model or not args.out:
        parser.error("--model and --out are required")

    arms = tuple(args.arm) if args.arm else ARMS
    delays = tuple(int(d) for d in args.delays.split(","))
    models = _parse_models(args.model)
    for name, (head, _) in models.items():
        for arm in arms:
            if arm == "rtc" and head != "flow":
                print(f"[skip] {name}: arm 'rtc' not applicable to single-pass head {head!r}",
                      flush=True)
                continue
            _assert_arm_supported(head, arm)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(models, args.stats_path, args.horizon)
    manifest_path = out_dir / "evaluation_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("[FATAL] refusing mixed resume: manifest differs")
    _save_json(manifest, manifest_path)
    results_path = out_dir / "rtc_latency_spatial.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    if results.get("_manifest") not in (None, manifest):
        raise SystemExit("[FATAL] refusing mixed resume: result manifest differs")
    results["_manifest"] = manifest

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    import numpy as np
    import torch

    stats = torch.load(args.stats_path) if args.stats_path else LeRobotDatasetMetadata(DATASET).stats
    for name, (head, checkpoint) in models.items():
        policy, (preprocessor, postprocessor) = load_policy(head, checkpoint, stats)
        for arm in arms:
            if arm == "rtc" and head != "flow":
                continue
            _configure_rtc(policy, arm, args.horizon)
            for delay in delays:
                cell = f"{name}|{arm}|d{delay}"
                cell_result = results.setdefault(cell, {"per_task": {}, "average": None})
                for task_id in range(N_TASKS):
                    task = cell_result["per_task"].setdefault(str(task_id), {"episodes": []})
                    episodes = task["episodes"]
                    if [ep["seed"] for ep in episodes] != list(range(1000, 1000 + len(episodes))):
                        raise SystemExit(f"[FATAL] invalid resume data for {cell}/task{task_id}")
                    env = _make_env(task_id)
                    task.setdefault("task", env.task_description)
                    for trial in range(len(episodes), args.trials):
                        env.init_state_id = trial
                        assert getattr(env, "init_state_id", None) == trial
                        episodes.append(_rollout_async(
                            policy, preprocessor, postprocessor, env, env.task_description,
                            1000 + trial, arm=arm, delay=delay, horizon=args.horizon,
                            guidance_probe=(trial == 0 and task_id == 0)))
                        task["success_rate"] = float(np.mean([e["success"] for e in episodes]))
                        _save_json(results, results_path)
                        print(f"[{cell}] task{task_id} trial{trial:02d}: "
                              f"{'OK' if episodes[-1]['success'] else 'x'}", flush=True)
                    env.close()
                rates = [e["success_rate"] for e in cell_result["per_task"].values()]
                cell_result["average"] = float(np.mean(rates))
                _save_json(results, results_path)
                print(f"[{cell}] AVERAGE {cell_result['average']:.4f}", flush=True)
        del policy
        torch.cuda.empty_cache()
    print("[rtc-latency] DONE", flush=True)


if __name__ == "__main__":
    main()
