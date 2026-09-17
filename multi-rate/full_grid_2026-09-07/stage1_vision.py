#!/usr/bin/env python3
"""Train Stage 1 visual DP and evaluate paired arms on CloseSingleDoor.

Examples:
  python stage1_vision.py TurnOffSinkFaucet --mode train
  python stage1_vision.py CloseSingleDoor --mode eval --port 8766 --k 4 \
      --arms native,zoh,spline_satfix,bspline_eps_satfix,gripper_sync
Only load trusted checkpoints (torch/pickle). The bridge uses a separate env.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import torch

from dp_min import EMA, MinMax, VisualDiffusionPolicy
from robocasa_bridge import recv_msg, send_msg
from robocasa_vision_data import VisionDataset, EpisodeWindows, PAPER, split_demos, sha256_file
from download_robocasa_vision_data import DEFAULT_OUTPUT_DIR

FORMAT_VERSION = 1
ARCHITECTURE_FIELDS = {
    "proprio_dim", "act_dim", "camera_keys", "image_layouts", "embedding_dim",
    "backbone", "n_obs", "horizon", "n_action_steps", "T", "down_dims",
}
BUDGET_FIELDS = {"steps", "batch_size", "microbatch", "lr", "seed", "ddim_steps", "ema_decay"}
DECIMATED_TASK = "CloseSingleDoor"
DECIMATED_ROBOCASA_TASK = "RC-CloseSingleDoor"
DEFAULT_COMPARISON_ARMS = (
    "native", "zoh", "spline_satfix", "bspline_eps_satfix", "gripper_sync",
)


def train_vision(policy, windows, steps=15_000, bs=256, microbatch=16, lr=1e-4,
                 dev="cuda", seed=0, log_every=100):
    if min(steps, bs, microbatch) < 1:
        raise ValueError("steps and batch sizes must be positive")
    policy.to(dev).train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1., s / 500) * .5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ema = EMA(policy)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for step in range(1, steps + 1):
        indices = rng.integers(len(windows), size=bs)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.
        for start in range(0, bs, microbatch):
            batch_indices = indices[start:start + microbatch]
            state, images, actions = windows.batch(batch_indices)
            loss = policy.loss(torch.as_tensor(state, device=dev), images,
                               torch.as_tensor(actions, device=dev))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite visual DP loss")
            fraction = len(batch_indices) / bs
            (loss * fraction).backward()
            loss_sum += loss.item() * fraction
        opt.step()
        schedule.step()
        ema.update(policy)
        # Existing state-only EMA has no running stats. ResNet BatchNorm does:
        # copy its calibrated buffers, including integer num_batches_tracked.
        with torch.no_grad():
            for target, source in zip(ema.m.buffers(), policy.buffers()):
                target.copy_(source)
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"[train vision] step {step}/{steps} loss={loss_sum:.5f} "
                  f"elapsed_s={time.monotonic() - started:.1f}", flush=True)
    return ema.m


def validate_checkpoint(checkpoint, metadata):
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("incompatible checkpoint format")
    if checkpoint.get("data_metadata") != metadata:
        raise ValueError("checkpoint/data metadata mismatch (identity, modalities, or shapes)")
    arch = checkpoint.get("architecture")
    if not isinstance(arch, dict) or not ARCHITECTURE_FIELDS.issubset(arch):
        raise ValueError("checkpoint is missing policy architecture values")
    budget = checkpoint.get("budget")
    if not isinstance(budget, dict) or not BUDGET_FIELDS.issubset(budget):
        raise ValueError("checkpoint is missing training budget or seed")
    if not isinstance(checkpoint.get("ema_policy"), dict) or not checkpoint["ema_policy"]:
        raise ValueError("checkpoint is missing EMA policy weights")
    if (arch["proprio_dim"] != sum(metadata["proprio_widths"].values()) or
            arch["act_dim"] != metadata["act_dim"] or
            list(arch["camera_keys"]) != metadata["camera_keys"] or
            arch["image_layouts"] != metadata["image_layouts"]):
        raise ValueError("architecture disagrees with data modalities")
    if (arch["backbone"] not in ("resnet18", "tiny") or
            type(arch["embedding_dim"]) is not int or arch["embedding_dim"] < 1 or
            type(arch["T"]) is not int or arch["T"] < 1):
        raise ValueError("invalid visual architecture")
    if (not arch["down_dims"] or any(type(width) is not int or width < 1 for width in arch["down_dims"])):
        raise ValueError("invalid U-Net dimensions")
    if (arch["n_obs"], arch["horizon"], arch["n_action_steps"]) != (2, 16, 8):
        raise ValueError("Stage 1 requires n_obs=2, horizon=16, n_action_steps=8")
    if (any(type(budget[key]) is not int or budget[key] < 1
            for key in ("steps", "batch_size", "microbatch")) or
            type(budget["seed"]) is not int or
            not isinstance(budget["lr"], (int, float)) or budget["lr"] <= 0 or
            not isinstance(budget["ema_decay"], (int, float)) or
            not 0 <= budget["ema_decay"] < 1):
        raise ValueError("invalid training budget or seed")
    if budget["ddim_steps"] != 10:
        raise ValueError("Stage 1 requires DDIM-10")
    tr, ev = checkpoint["train_demo_indices"], checkpoint["eval_demo_indices"]
    if (not tr or not ev or set(tr) & set(ev) or len(set(tr + ev)) != len(tr + ev) or
            any(type(i) is not int or i < 0 or i >= len(metadata["demo_names"]) for i in tr + ev)):
        raise ValueError("checkpoint has invalid or overlapping demo split")
    for key, width in (("state_normalizer", arch["proprio_dim"]), ("action_normalizer", arch["act_dim"])):
        norm = checkpoint[key]
        for field in ("lo", "hi", "rng"):
            value = np.asarray(norm[field])
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"invalid {key}/{field}")
        span = np.asarray(norm["hi"]) - np.asarray(norm["lo"])
        if np.any(span < 0) or not np.allclose(norm["rng"], np.where(span < 1e-8, 1., span)):
            raise ValueError(f"invalid {key} range")


def default_max_steps(task):
    return 300 if task == "CoffeePressButton" else 500


def experiment_request(task, metadata, n_train, n_eval, steps, batch_size, microbatch,
                       training_seed, eval_seed, max_steps=None):
    """Canonical requested experiment identity used by direct eval and campaigns."""
    if metadata.get("task") != task:
        raise ValueError(f"requested task {task} does not match dataset task {metadata.get('task')}")
    train_indices, eval_indices = split_demos(len(metadata["demo_names"]), n_train, n_eval)
    requested = {
        "task": task,
        "data_sha256": metadata["data_sha256"],
        "train_demo_indices": train_indices,
        "eval_demo_indices": eval_indices,
        "steps": steps,
        "batch_size": batch_size,
        "microbatch": microbatch,
        "training_seed": training_seed,
        "eval_seed": eval_seed,
        "max_steps": default_max_steps(task) if max_steps is None else max_steps,
    }
    if (any(type(requested[key]) is not int or requested[key] < 1
            for key in ("steps", "batch_size", "microbatch", "max_steps")) or
            type(training_seed) is not int or training_seed < 0 or
            type(eval_seed) is not int or eval_seed < 0):
        raise ValueError("requested budget and seeds must be valid integers")
    return requested


def validate_checkpoint_request(checkpoint, metadata, requested):
    """Reject reuse when a checkpoint represents a different requested experiment."""
    validate_checkpoint(checkpoint, metadata)
    mismatches = []
    if requested["task"] != metadata["task"]:
        mismatches.append("task")
    if requested["data_sha256"] != metadata["data_sha256"]:
        mismatches.append("dataset hash")
    if checkpoint["train_demo_indices"] != requested["train_demo_indices"]:
        mismatches.append("training split/count")
    if checkpoint["eval_demo_indices"] != requested["eval_demo_indices"]:
        mismatches.append("evaluation split/count")
    budget = checkpoint["budget"]
    for checkpoint_key, request_key in (
        ("steps", "steps"), ("batch_size", "batch_size"), ("microbatch", "microbatch"),
        ("seed", "training_seed"),
    ):
        if budget[checkpoint_key] != requested[request_key]:
            mismatches.append(request_key.replace("_", " "))
    if mismatches:
        raise ValueError("checkpoint does not match requested experiment: " + ", ".join(mismatches))


def validate_result_request(result, checkpoint_path, data, checkpoint, requested):
    """Validate a complete or partial JSON record before campaign reuse."""
    if not isinstance(result, dict):
        raise ValueError("result JSON payload must be an object")
    checkpoint_path = Path(checkpoint_path).resolve()
    expected_checkpoint_hash = sha256_file(checkpoint_path)
    mismatches = []
    if result.get("stage") != 1:
        mismatches.append("stage")
    if result.get("task") != requested["task"]:
        mismatches.append("task")
    task_mapping = result.get("task_mapping")
    if not isinstance(task_mapping, dict) or task_mapping.get("dataset_task") != requested["task"]:
        mismatches.append("task mapping")
    if result.get("checkpoint_path") != str(checkpoint_path):
        mismatches.append("checkpoint path")
    if result.get("checkpoint_sha256") != expected_checkpoint_hash:
        mismatches.append("checkpoint hash")
    if result.get("data_path") != str(Path(data.path).resolve()):
        mismatches.append("data path")
    if result.get("data_metadata") != data.metadata:
        mismatches.append("dataset identity/hash")
    if result.get("train_demo_indices") != requested["train_demo_indices"]:
        mismatches.append("training split/count")
    if result.get("eval_demo_indices") != requested["eval_demo_indices"]:
        mismatches.append("evaluation split/count")
    if result.get("training_budget") != checkpoint["budget"]:
        mismatches.append("training budget")
    if result.get("eval_seed") != requested["eval_seed"]:
        mismatches.append("evaluation seed")
    if result.get("max_steps") != requested["max_steps"]:
        mismatches.append("evaluation max steps")
    episodes = result.get("episodes")
    episode_count = result.get("episode_count")
    if type(result.get("complete")) is not bool:
        mismatches.append("completion status")
    if (type(episode_count) is not int or episode_count < 0 or
            not isinstance(episodes, list) or len(episodes) != episode_count):
        mismatches.append("episode count/records")
    if result.get("complete") is True and (
            type(episode_count) is not int or episode_count < len(requested["eval_demo_indices"])):
        mismatches.append("at-least-requested episode count")
    if mismatches:
        raise ValueError("result does not match requested experiment: " + ", ".join(mismatches))


def campaign_artifact_status(checkpoint_path, result_path, data, requested):
    """Return train-eval/eval/skip, rejecting stale artifacts rather than overwriting them."""
    checkpoint_path, result_path = Path(checkpoint_path), Path(result_path)
    if not checkpoint_path.exists():
        if result_path.exists():
            raise ValueError("result exists without its checkpoint; use a distinct output directory")
        return "train-eval"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError(f"cannot reuse checkpoint: {exc}") from exc
    validate_checkpoint_request(checkpoint, data.metadata, requested)
    if not result_path.exists():
        return "eval"
    try:
        result = json.loads(result_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot reuse result JSON: {exc}") from exc
    validate_result_request(result, checkpoint_path, data, checkpoint, requested)
    return "skip" if result.get("complete") is True else "eval"


def save_checkpoint(path, policy, architecture, data, state_norm, action_norm, train_indices, eval_indices, budget):
    checkpoint = {
        "format_version": FORMAT_VERSION, "ema_policy": policy.state_dict(),
        "architecture": architecture, "data_metadata": data.metadata, "data_path": str(data.path),
        "state_normalizer": state_norm.state(), "action_normalizer": action_norm.state(),
        "train_demo_indices": train_indices, "eval_demo_indices": eval_indices, "budget": budget,
    }
    validate_checkpoint(checkpoint, data.metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, metadata, dev, requested=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if requested is None:
        validate_checkpoint(checkpoint, metadata)
    else:
        validate_checkpoint_request(checkpoint, metadata, requested)
    policy = VisualDiffusionPolicy(**checkpoint["architecture"])
    policy.load_state_dict(checkpoint["ema_policy"], strict=True)
    return (policy.to(dev).eval(), MinMax.from_state(checkpoint["state_normalizer"]),
            MinMax.from_state(checkpoint["action_normalizer"]), checkpoint)


class VisionClient:
    def __init__(self, task, port):
        self.task = task
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=600)

    def request(self, cmd, **kwargs):
        send_msg(self.sock, {"cmd": cmd, "task": self.task, **kwargs})
        reply = recv_msg(self.sock)
        if reply is None:
            raise ConnectionError("vision bridge disconnected")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def close(self):
        try:
            try:
                self.request("close")
            except (ConnectionError, OSError, RuntimeError):
                # Preserve an in-flight evaluation error if the bridge also
                # disappeared before it could acknowledge client cleanup.
                pass
        finally:
            self.sock.close()


def native_rollout(client, policy, state_norm, action_norm, demo_index, max_steps, dev):
    observation = client.request("reset_to", idx=demo_index)["obs"]
    history = deque([observation] * policy.n_obs, maxlen=policy.n_obs)
    executed, success = 0, False
    while executed < max_steps:
        states = np.stack([o["state"] for o in history])[None]
        images = {k: np.stack([o["images"][k] for o in history])[None] for k in policy.camera_keys}
        chunk = policy.sample(torch.as_tensor(state_norm.norm(states), device=dev), images, n_steps=10)
        actions = action_norm.denorm(chunk[0].cpu().numpy())
        # Native control: first 8 chronological actions, one env step per action.
        for action in actions[:policy.n_action_steps]:
            reply = client.request("step", action=action)
            executed += 1
            history.append(reply["obs"])
            success = success or reply["success"]
            if success or reply["done"] or executed >= max_steps:
                return {"demo_index": demo_index, "success": bool(success), "steps": executed}
    return {"demo_index": demo_index, "success": bool(success), "steps": executed}


def apply_comparison_arm(chunk, arm, n_hold, k):
    """Delegate comparison-arm transformation to the validated harness wiring."""
    import harness

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    previous_k = harness.K
    try:
        harness.K = k
        return harness.apply_arm(chunk, arm, n_hold)
    finally:
        harness.K = previous_k


def decimated_rollout(client, policy, state_norm, action_norm, demo_index, ei,
                      max_steps, dev, arm, k, n_hold):
    """Run one paired visual-policy arm with harness-identical chunk execution."""
    observation = client.request("reset_to", idx=demo_index)["obs"]
    history = deque([observation] * policy.n_obs, maxlen=policy.n_obs)
    executed, success, replan = 0, False, 0
    while executed < max_steps:
        states = np.stack([o["state"] for o in history])[None]
        images = {key: np.stack([o["images"][key] for o in history])[None]
                  for key in policy.camera_keys}
        torch.manual_seed(1_000_003 * ei + replan)  # paired noise across arms
        predicted = policy.sample(
            torch.as_tensor(state_norm.norm(states), device=dev), images, n_steps=10)
        actions = action_norm.denorm(predicted[0].cpu().numpy())
        chunk = np.clip(actions[:policy.n_action_steps], -1, 1).astype(np.float32)
        exec_chunk = apply_comparison_arm(chunk, arm, n_hold, k)
        for action in exec_chunk:
            reply = client.request("step", action=action)
            executed += 1
            history.append(reply["obs"])
            success = success or reply["success"]
            if success or reply["done"] or executed >= max_steps:
                return {"demo_index": demo_index, "success": bool(success), "steps": executed}
        replan += 1
    return {"demo_index": demo_index, "success": bool(success), "steps": executed}


def comparison_result(task, episodes, checkpoint_path, data, checkpoint, max_steps, eval_seed):
    if not episodes:
        raise ValueError("at least one evaluation episode required")
    label, reference = PAPER[task]
    successes = sum(bool(e["success"]) for e in episodes)
    rate = 100. * successes / len(episodes)
    return {
        "stage": 1, "task": task, "paper_task": label, "policy": "visual_diffusion_policy",
        "task_mapping": {"dataset_task": task, "paper_task": label},
        "k": 1, "action_execution": "first 8 predicted actions directly; no decimation/resampling",
        "ddim_steps": 10, "episode_count": len(episodes), "success_count": successes,
        "success_percent": rate, "paper_diff_1x_base_percent": reference,
        "gap_percentage_points": rate - reference, "episodes": episodes,
        "max_steps": max_steps, "eval_seed": eval_seed,
        "checkpoint_path": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_format_version": checkpoint["format_version"],
        "policy_architecture": checkpoint["architecture"],
        "data_path": str(data.path), "data_metadata": data.metadata,
        "train_demo_indices": checkpoint["train_demo_indices"],
        "eval_demo_indices": checkpoint["eval_demo_indices"], "training_budget": checkpoint["budget"],
        "interpretation": "If results remain far below the paper, demo count is the next likely cause. "
                          "Stage 1 uses 35 training demos by default. Stage 2 is not started.",
        "scientific_limit": "One task is not a four-task validation; fewer than 15 episodes is a smoke evaluation.",
    }


def decimated_comparison_result(task, episodes_by_arm, checkpoint_path, data, checkpoint,
                                max_steps, eval_seed, k, n_hold):
    """Build a harness-style paired multi-arm result with exact McNemar contrasts."""
    from harness import exact_mcnemar

    arms = list(episodes_by_arm)
    if not arms:
        raise ValueError("at least one comparison arm required")
    episode_count = len(episodes_by_arm[arms[0]])
    if episode_count < 1 or any(len(episodes_by_arm[arm]) != episode_count for arm in arms):
        raise ValueError("all arms must contain the same positive number of episodes")
    demo_indices = [[episode["demo_index"] for episode in episodes_by_arm[arm]] for arm in arms]
    if any(indices != demo_indices[0] for indices in demo_indices[1:]):
        raise ValueError("paired arms must use identical demo indices in identical order")

    success = {
        arm: [bool(episode["success"]) for episode in episodes_by_arm[arm]] for arm in arms
    }
    step_counts = {
        arm: [int(episode["steps"]) for episode in episodes_by_arm[arm]] for arm in arms
    }
    success_arrays = {arm: np.asarray(success[arm], dtype=bool) for arm in arms}
    contrasts = {}
    for arm in arms:
        contrasts[arm] = {}
        references = ["zoh", "native"]
        if arm == "gripper_sync":
            references.extend(["spline_satfix", "bspline_eps_satfix"])
        for reference in references:
            if arm == reference or reference not in success_arrays:
                continue
            arm_only, reference_only, p_value = exact_mcnemar(
                success_arrays[arm], success_arrays[reference])
            contrasts[arm][f"vs_{reference}"] = {
                "delta_pp": 100.0 * (
                    success_arrays[arm].mean() - success_arrays[reference].mean()),
                "p": p_value,
                "arm_only": arm_only,
                "reference_only": reference_only,
            }

    label, reference = PAPER[task]
    success_counts = {arm: int(success_arrays[arm].sum()) for arm in arms}
    success_percent = {
        arm: 100.0 * success_counts[arm] / episode_count for arm in arms
    }
    return {
        "stage": 1,
        "task": task,
        "paper_task": label,
        "policy": "visual_diffusion_policy",
        "task_mapping": {"dataset_task": task, "paper_task": label},
        "k": k,
        "n_hold": n_hold,
        "action_execution": "clip each predicted 8-step chunk to [-1,1], then apply harness.apply_arm",
        "ddim_steps": 10,
        "arms": arms,
        "episode_count": episode_count,
        "demo_indices": demo_indices[0],
        "success": success,
        "step_counts": step_counts,
        "episodes": episodes_by_arm,
        "success_count": success_counts,
        "success_percent": success_percent,
        "contrasts": contrasts,
        "paper_diff_1x_base_percent": reference,
        "max_steps": max_steps,
        "eval_seed": eval_seed,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_format_version": checkpoint["format_version"],
        "policy_architecture": checkpoint["architecture"],
        "data_path": str(data.path),
        "data_metadata": data.metadata,
        "train_demo_indices": checkpoint["train_demo_indices"],
        "eval_demo_indices": checkpoint["eval_demo_indices"],
        "training_budget": checkpoint["budget"],
        "scientific_limit": "Exact McNemar p-values are reported unstarred; n=15 is not a significance claim.",
    }


def parse_arms(value):
    arms = tuple(arm.strip() for arm in value.split(",") if arm.strip())
    if not arms:
        raise argparse.ArgumentTypeError("--arms must select at least one arm")
    if len(set(arms)) != len(arms):
        raise argparse.ArgumentTypeError("--arms must not contain duplicates")
    unknown = [arm for arm in arms if arm not in DEFAULT_COMPARISON_ARMS]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unsupported arm(s): " + ", ".join(unknown) +
            "; choose from " + ", ".join(DEFAULT_COMPARISON_ARMS))
    return arms


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=PAPER)
    parser.add_argument("--mode", choices=("train", "eval", "train-eval"), default="train-eval")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("stage1_vision_results"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=16,
                        help="gradient accumulation; effective batch stays 256")
    parser.add_argument("--n-train", type=int, default=35)
    parser.add_argument("--n-eval", type=int, default=15, help="held-out split; eval reuse must match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int,
                        help="default: existing native limits, coffee=300, others=500")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--arms", type=parse_arms,
                        help="comma list; default all comparison arms for CloseSingleDoor")
    parser.add_argument("--check-artifacts", action="store_true",
                        help="print campaign reuse status without training, evaluation, or bridge access")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.k < 1:
        parser.error("--k must be positive")
    if args.mode != "train" and not args.check_artifacts:
        if args.task != DECIMATED_TASK and args.arms is not None:
            parser.error("--arms is supported for CloseSingleDoor evaluation only")
        if args.task == DECIMATED_TASK:
            import harness

            expected_max_steps = harness.ROBOCASA[DECIMATED_ROBOCASA_TASK]["max_steps"]
            if args.max_steps is not None and args.max_steps != expected_max_steps:
                parser.error(f"CloseSingleDoor comparison requires --max-steps {expected_max_steps}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.checkpoint or args.output_dir / f"{args.task}.pt"
    data = VisionDataset(args.task, args.data_dir)
    try:
        requested = experiment_request(
            args.task, data.metadata, args.n_train, args.n_eval, args.steps, args.batch_size,
            args.microbatch, args.seed, args.eval_seed, args.max_steps,
        )
        if args.check_artifacts:
            result_path = args.output_dir / f"{args.task}_native.json"
            try:
                status = campaign_artifact_status(path, result_path, data, requested)
            except ValueError as exc:
                parser.error(f"incompatible existing Stage 1 artifact: {exc}")
            print(status)
            return
        if args.mode != "eval":
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            train_indices, eval_indices = split_demos(len(data.demos), args.n_train, args.n_eval)
            print(f"[data] {args.task} train={train_indices} eval={eval_indices} cameras={data.camera_keys}", flush=True)
            episodes = [data.episode(i) for i in train_indices]
            state_norm = MinMax(np.concatenate([e["state"] for e in episodes]))
            action_norm = MinMax(np.concatenate([e["actions"] for e in episodes]))
            architecture = {"proprio_dim": sum(data.widths.values()), "act_dim": data.act_dim,
                            "camera_keys": list(data.camera_keys), "image_layouts": data.layouts,
                            "embedding_dim": 64, "backbone": "resnet18", "n_obs": 2, "horizon": 16,
                            "n_action_steps": 8, "T": 100, "down_dims": (256, 512, 1024)}
            windows = EpisodeWindows(episodes, state_norm, action_norm)
            policy = VisualDiffusionPolicy(**architecture)
            ema = train_vision(policy, windows, args.steps, args.batch_size, args.microbatch,
                               dev=args.device, seed=args.seed)
            budget = {"steps": args.steps, "batch_size": args.batch_size, "microbatch": args.microbatch,
                      "lr": 1e-4, "seed": args.seed, "ddim_steps": 10, "ema_decay": .9999}
            save_checkpoint(path, ema, architecture, data, state_norm, action_norm, train_indices, eval_indices, budget)
            print(f"[checkpoint] saved EMA policy: {path}", flush=True)
            del policy, ema, windows, episodes
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if args.mode != "train":
            policy, state_norm, action_norm, checkpoint = load_checkpoint(
                path, data.metadata, args.device, requested=requested)
            torch.manual_seed(args.eval_seed)
            np.random.seed(args.eval_seed)
            client = VisionClient(args.task, args.port)
            if args.task == DECIMATED_TASK:
                import harness

                arms = args.arms or DEFAULT_COMPARISON_ARMS
                episodes_by_arm = {arm: [] for arm in arms}
                robocasa_config = harness.ROBOCASA[DECIMATED_ROBOCASA_TASK]
                max_steps = robocasa_config["max_steps"]
                n_hold = robocasa_config["n_hold"]
                try:
                    if client.request("metadata")["metadata"] != data.metadata:
                        raise ValueError("bridge/client dataset metadata mismatch")
                    for episode_index, index in enumerate(checkpoint["eval_demo_indices"]):
                        for arm in arms:
                            result = decimated_rollout(
                                client, policy, state_norm, action_norm, index, episode_index,
                                max_steps, args.device, arm, args.k, n_hold)
                            episodes_by_arm[arm].append(result)
                        counts = " ".join(
                            f"{arm}={sum(episode['success'] for episode in episodes_by_arm[arm])}"
                            for arm in arms)
                        print(f"[eval] {args.task} {episode_index + 1}/"
                              f"{len(checkpoint['eval_demo_indices'])} {counts}", flush=True)
                        report = decimated_comparison_result(
                            args.task, episodes_by_arm, path, data, checkpoint, max_steps,
                            args.eval_seed, args.k, n_hold)
                        report["complete"] = (
                            episode_index + 1 == len(checkpoint["eval_demo_indices"]))
                        result_path = args.output_dir / f"{args.task}_decimated_k{args.k}.json"
                        temporary = result_path.with_suffix(".json.partial")
                        temporary.write_text(json.dumps(report, indent=2) + "\n")
                        temporary.replace(result_path)
                finally:
                    client.close()
                print(f"\n=== {args.task} visual DP, k={args.k}, n={report['episode_count']} ===")
                for arm in arms:
                    line = (f"  {arm:22s} {report['success_percent'][arm]:6.1f}% "
                            f"({report['success_count'][arm]}/{report['episode_count']})")
                    for contrast, values in report["contrasts"][arm].items():
                        reference = contrast.removeprefix("vs_")
                        line += (f"  | vs {reference}: {values['delta_pp']:+6.1f}pp "
                                 f"p={values['p']:.3g}")
                    print(line)
                print(f"[saved] {result_path}", flush=True)
            else:
                episodes = []
                max_steps = requested["max_steps"]
                try:
                    if client.request("metadata")["metadata"] != data.metadata:
                        raise ValueError("bridge/client dataset metadata mismatch")
                    for i, index in enumerate(checkpoint["eval_demo_indices"]):
                        result = native_rollout(
                            client, policy, state_norm, action_norm, index, max_steps, args.device)
                        episodes.append(result)
                        print(f"[eval] {args.task} {i + 1}/"
                              f"{len(checkpoint['eval_demo_indices'])} "
                              f"success={result['success']} steps={result['steps']}", flush=True)
                        report = comparison_result(
                            args.task, episodes, path, data, checkpoint, max_steps, args.eval_seed)
                        report["complete"] = len(episodes) == len(checkpoint["eval_demo_indices"])
                        result_path = args.output_dir / f"{args.task}_native.json"
                        temporary = result_path.with_suffix(".json.partial")
                        temporary.write_text(json.dumps(report, indent=2) + "\n")
                        temporary.replace(result_path)
                finally:
                    client.close()
                print(f"[paper] {report['paper_task']}: "
                      f"{report['success_count']}/{report['episode_count']} "
                      f"({report['success_percent']:.1f}%) vs "
                      f"{report['paper_diff_1x_base_percent']:.0f}%; "
                      f"gap {report['gap_percentage_points']:+.1f} percentage points. "
                      f"{report['interpretation']}", flush=True)
    finally:
        data.close()


if __name__ == "__main__":
    main()
