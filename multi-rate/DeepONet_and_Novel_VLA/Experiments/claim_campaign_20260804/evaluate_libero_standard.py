"""Locked, native-rate LIBERO-Spatial evaluation (50 trials per task, no videos)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SUITE = "libero_spatial"
N_TASKS = 10
TRIALS_PER_TASK = 50
REPLAN = 1
MAX_STEPS = 220
CONTROL_FREQ = 20
DATASET = "lerobot/libero_spatial_image"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def resolve_latest(checkpoint: str) -> str:
    path = Path(checkpoint)
    if path.name in ("LATEST", "BEST"):
        pointer = path.parent / f"{path.name}.txt"
        if not pointer.exists() and path.name == "BEST":
            pointer = path.parent / "LATEST.txt"
        if pointer.exists():
            return str(path.parent / pointer.read_text().strip())
    return str(path)


def _locked_protocol() -> dict:
    return {
        "suite": SUITE,
        "n_tasks": N_TASKS,
        "trials_per_task": TRIALS_PER_TASK,
        "replan": REPLAN,
        "max_steps": MAX_STEPS,
        "control_freq": CONTROL_FREQ,
        "rollout_reset_seed": "1000 + trial_index",
        "videos": False,
    }


def _assert_native_runtime() -> None:
    expected = {
        "DEEPONET_QUERY_FACTOR": "1",
        "DEEPONET_QUERY_POINTS": "0",
        "DEEPONET_FOLD_ALIASES": "0",
        "DEEPONET_GAAR": "0",
        "DEEPONET_RETURN_QUERY_GRID": "0",
        "DEEPONET_ACTION_INTERP_FACTOR": "1",
        "TEMPO_SPEED": "1.0",
    }
    bad = {key: os.environ[key] for key, value in expected.items()
           if key in os.environ and os.environ[key] != value}
    if bad:
        raise SystemExit(f"[FATAL] standard evaluator rejects non-native runtime overrides: {bad}")
    rate = os.environ.get("DEEPONET_RATE_HZ", str(CONTROL_FREQ))
    if float(rate) != CONTROL_FREQ:
        raise SystemExit(f"[FATAL] standard evaluator requires DEEPONET_RATE_HZ={CONTROL_FREQ}")


def _parse_models(specs: list[str]) -> dict[str, tuple[str, str]]:
    models = {}
    for spec in specs:
        try:
            name, head, checkpoint = spec.split("=", 2)
        except ValueError as exc:
            raise SystemExit(f"[FATAL] invalid --model {spec!r}; expected NAME=HEAD=CKPT") from exc
        if not name or head not in {"flow", "deeponet"} or not checkpoint or name in models:
            raise SystemExit(f"[FATAL] invalid or duplicate --model {spec!r}")
        models[name] = (head, resolve_latest(checkpoint))
    return models


def _verify_deeponet_provenance(checkpoint: str) -> None:
    run_config = Path(checkpoint).parent.parent / "run_config.json"
    if not run_config.is_file():
        raise SystemExit(f"[FATAL] DeepONet checkpoint lacks run_config.json: {run_config}")
    try:
        trained_head = json.loads(run_config.read_text())["args"].get("deeponet_head", "deeponet")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(f"[FATAL] invalid training provenance: {run_config}") from exc
    runtime_head = os.environ.get("DEEPONET_HEAD", "deeponet")
    if trained_head != runtime_head:
        raise SystemExit(f"[FATAL] eval/train head mismatch: trained {trained_head}, runtime {runtime_head}")


def load_policy(head: str, checkpoint: str, dataset_stats):
    import safetensors.torch as safetensors_torch
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    if head == "flow":
        from modeling_smolvla_ph import SmolVLAPHPolicy
        policy = SmolVLAPHPolicy.from_pretrained(checkpoint, ph_enabled=False)
    else:
        _verify_deeponet_provenance(checkpoint)
        from modeling_smolvla_deeponet_v2 import SmolVLADeepONetPolicy
        policy = SmolVLADeepONetPolicy.from_pretrained(
            checkpoint,
            ph_enabled=False,
            deeponet_p=int(os.environ.get("DEEPONET_P", 256)),
            deeponet_blocks=int(os.environ.get("DEEPONET_BLOCKS", 3)),
            deeponet_queries=int(os.environ.get("DEEPONET_QUERIES", 8)),
            deeponet_fourier=int(os.environ.get("DEEPONET_FOURIER", 16)),
            deeponet_head=os.environ.get("DEEPONET_HEAD", "deeponet"),
            deeponet_pool_norm=bool(int(os.environ.get("DEEPONET_POOL_CHANNEL_NORM", "0"))),
            deeponet_trunk_bandlimit=bool(int(os.environ.get("DEEPONET_TRUNK_BANDLIMIT", "0"))),
        )
        stored, built = {}, {key: tuple(value.shape) for key, value in policy.state_dict().items()
                             if "deeponet" in key}
        for path in Path(checkpoint).glob("model*.safetensors"):
            with safetensors_torch.safe_open(path, framework="pt") as handle:
                stored.update({key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()
                               if "deeponet" in key})
        if stored != built:
            raise SystemExit("[FATAL] model head and checkpoint tensors differ; refusing random init")
        if os.environ.get("DEEPONET_HEAD") in {"ti", "til", "asrc"}:
            from rate_integrated_deeponet import RateIntegratedDeepONetHead
            heads = [module for module in policy.modules() if isinstance(module, RateIntegratedDeepONetHead)]
            if len(heads) != 1:
                raise SystemExit(f"[FATAL] expected one rate-integrated head, found {len(heads)}")
            heads[0].configure_action_stats(dataset_stats["action"]["mean"], dataset_stats["action"]["std"])
            heads[0].set_rate(CONTROL_FREQ)
        history_steps = int(os.environ.get("DEEPONET_STATE_HISTORY_STEPS", "1"))
        if history_steps not in {1, 8}:
            raise SystemExit("[FATAL] DEEPONET_STATE_HISTORY_STEPS must be 1 or 8")
        if history_steps == 8:
            policy.configure_state_history(history_steps)
    policy = policy.to("cuda").eval()
    policy.config.n_action_steps = REPLAN
    return policy, make_smolvla_pre_post_processors(policy.config, dataset_stats=dataset_stats)


def _policy_input(obs, task_description):
    import numpy as np
    import torch

    def image(value):
        tensor = torch.as_tensor(np.asarray(value)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return torch.flip(tensor, dims=[2, 3])

    state = obs["robot_state"]
    quat = torch.as_tensor(np.asarray(state["eef"]["quat"])).float().reshape(1, 4)
    w = quat[:, 3].clamp(-1.0, 1.0)
    denominator = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    axis_angle = torch.zeros((1, 3))
    if denominator.item() > 1e-10:
        axis_angle = quat[:, :3] / denominator.unsqueeze(1) * (2.0 * torch.acos(w)).unsqueeze(1)
    robot_state = torch.cat((
        torch.as_tensor(np.asarray(state["eef"]["pos"])).float().reshape(1, 3),
        axis_angle,
        torch.as_tensor(np.asarray(state["gripper"]["qpos"])).float().reshape(1, 2),
    ), dim=-1)
    return {
        "observation.images.image": image(obs["pixels"]["image"]),
        "observation.images.wrist_image": image(obs["pixels"]["image2"]),
        "observation.state": robot_state,
        "task": [task_description],
    }


def _make_env(task_id: int):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    return LiberoEnv(task_suite=suite, task_suite_name=SUITE, task_id=task_id,
                     obs_type="pixels_agent_pos", control_mode="relative",
                     observation_height=256, observation_width=256)


def _rollout(policy, preprocessor, postprocessor, env, task_description: str, seed: int) -> dict:
    import torch

    policy.reset()
    observation, _ = env.reset(seed=seed)
    for step in range(1, MAX_STEPS + 1):
        batch = preprocessor(_policy_input(observation, task_description))
        batch = {key: value.to("cuda") if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch)
        observation, _, terminated, truncated, info = env.step(
            postprocessor(action).to("cpu").float().numpy().reshape(-1))
        if info.get("is_success", False):
            return {"seed": seed, "success": True, "steps": step}
        if terminated or truncated:
            break
    return {"seed": seed, "success": False, "steps": step}


def _manifest(models: dict[str, tuple[str, str]], stats_path: str | None) -> dict:
    root = Path(__file__).resolve().parent
    files = ("evaluate_libero_standard.py", "modeling_smolvla_deeponet_v2.py",
             "deeponet_head_v2.py", "rate_integrated_deeponet.py", "tempo_vsta.py",
             "state_history.py", "ncde_style_deeponet.py")
    return {
        "protocol": _locked_protocol(),
        "dataset": DATASET,
        "stats_path": str(Path(stats_path).resolve()) if stats_path else None,
        "stats_sha256": _optional_sha256(Path(stats_path)) if stats_path else None,
        "code_sha256": {name: _optional_sha256(root / name) for name in files},
        "models": {name: {"head": head, "checkpoint": str(Path(checkpoint).resolve()),
                            "model_sha256": _sha256(Path(checkpoint) / "model.safetensors")}
                   for name, (head, checkpoint) in models.items()},
        "runtime": {key: os.environ.get(key) for key in sorted(os.environ)
                    if key.startswith("DEEPONET_") or key == "TEMPO_SPEED"},
    }


def _save_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _selfcheck() -> None:
    assert _locked_protocol()["replan"] == 1
    assert _locked_protocol()["max_steps"] == 220
    assert _parse_models(["a=flow=/tmp/a", "b=deeponet=/tmp/b"])["b"][0] == "deeponet"
    print("SELF_CHECK_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="NAME=flow|deeponet=CHECKPOINT")
    parser.add_argument("--out")
    parser.add_argument("--stats_path")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if not args.model:
        parser.error("at least one --model is required")
    if not args.out:
        parser.error("--out is required")
    _assert_native_runtime()
    models = _parse_models(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(models, args.stats_path)
    manifest_path = out_dir / "evaluation_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("[FATAL] refusing mixed resume: manifest differs")
    _save_json(manifest, manifest_path)
    results_path = out_dir / "standard_libero_spatial.json"
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
        model_result = results.setdefault(name, {"per_task": {}, "average": None})
        for task_id in range(N_TASKS):
            task = model_result["per_task"].setdefault(str(task_id), {"episodes": []})
            episodes = task["episodes"]
            if len(episodes) > TRIALS_PER_TASK or [ep["seed"] for ep in episodes] != list(range(1000, 1000 + len(episodes))):
                raise SystemExit(f"[FATAL] invalid resume data for {name}/task{task_id}")
            env = _make_env(task_id)
            task.setdefault("task", env.task_description)
            for trial in range(len(episodes), TRIALS_PER_TASK):
                env.init_state_id = trial
                assert getattr(env, "init_state_id", None) == trial
                episode = _rollout(policy, preprocessor, postprocessor, env, env.task_description, 1000 + trial)
                episodes.append(episode)
                task["success_rate"] = float(np.mean([item["success"] for item in episodes]))
                _save_json(results, results_path)
                print(f"[{name}] task{task_id} trial{trial:02d}: {'OK' if episode['success'] else 'x'}", flush=True)
            env.close()
        rates = [entry["success_rate"] for entry in model_result["per_task"].values()]
        model_result["average"] = float(np.mean(rates))
        _save_json(results, results_path)
        del policy
        torch.cuda.empty_cache()
    print("[standard] DONE", flush=True)


if __name__ == "__main__":
    main()
