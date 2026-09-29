"""
Evaluate multi-rate resamplers on gym-pusht using the official Hub checkpoint (lerobot/diffusion_pusht).
Evaluates absolute-position action chunks by converting them to displacements from the
current agent position, resampling those displacements, and integrating back to positions.
Evaluates:
  - native (1X / unresampled baseline)
  - zoh
  - spline
  - tac_fold
  - ctac_position
  - bspline_eps_raw

Strictly paired across fixed eval seeds (e.g. 1000..1049 or 1000..1099).
"""
import argparse, json, os, sys, time
import numpy as np, torch
from scipy.optimize import minimize, LinearConstraint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from resample_math import decimate_and_resample, RESAMPLERS
from resample_bspline2 import resample_bspline_eps

RESAMPLERS['bspline_eps_raw'] = lambda b, k: resample_bspline_eps(b, k, eps=0.005)
from resample_qp import resample_qp_anchor
RESAMPLERS['qp_anchor'] = resample_qp_anchor  # prereg comparator (as harness.py registers it)


def resample_ctac_position(deltas: np.ndarray, anchor: np.ndarray, k: int) -> np.ndarray:
    """TAC-anchored QP with bounds on absolute positions, not on deltas."""
    n_blocks, d = len(deltas) // k, deltas.shape[1]
    block_sum = deltas[:n_blocks * k].reshape(n_blocks, k, d).sum(axis=1)
    tac = RESAMPLERS["tac_fold"](block_sum, k).astype(np.float64)
    T = n_blocks * k
    A = np.kron(np.eye(n_blocks), np.ones((1, k)))
    L = np.tril(np.ones((T, T)))
    out = np.empty_like(tac)
    for dim in range(d):
        target = block_sum[:, dim].astype(np.float64)
        x0 = np.repeat(target / k, k)
        constraints = [
            LinearConstraint(A, target, target),
            LinearConstraint(L, -1.0 - anchor[dim], 1.0 - anchor[dim]),
        ]
        da = np.diff(tac[:, dim])

        def objective(v):
            return float(np.sum((np.diff(v) - da) ** 2))

        def gradient(v):
            error = np.diff(v) - da
            g = np.zeros_like(v)
            g[:-1] -= 2 * error
            g[1:] += 2 * error
            return g

        result = minimize(objective, x0, jac=gradient, method="SLSQP",
                          constraints=constraints, options={"maxiter": 200, "ftol": 1e-9})
        if not result.success:
            raise RuntimeError(f"C-TAC position QP failed: {result.message}")
        out[:, dim] = result.x
    return out.astype(np.float32)

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.envs.factory import make_env
from lerobot.configs.types import FeatureType
from lerobot.envs.configs import PushtEnv
from lerobot.scripts.lerobot_eval import eval_policy


def resample_chunk(chunk_np: np.ndarray, anchor_np: np.ndarray, k: int, arm: str) -> np.ndarray:
    """Resample absolute targets through their displacement path.

    ``anchor_np`` is the current agent position expressed in the action normalizer's
    coordinates. Exact resamplers therefore preserve every coarse block endpoint.
    """
    if arm == "native" or k == 1:
        return chunk_np.copy()
    deltas = np.diff(np.concatenate([anchor_np[None], chunk_np], axis=0), axis=0)
    reconstructed = (resample_ctac_position(deltas, anchor_np, k) if arm == "ctac_position"
                     else decimate_and_resample(deltas, k, arm))
    return anchor_np[None] + np.cumsum(reconstructed, axis=0)


def state_norm_to_action_norm(state, state_min, state_max, action_min, action_max):
    """Move normalized agent positions between the state and action MIN_MAX frames."""
    raw = (state + 1.0) * 0.5 * (state_max - state_min) + state_min
    return 2.0 * (raw - action_min) / (action_max - action_min) - 1.0


def self_check():
    anchor = np.array([-0.4, 0.2], dtype=np.float32)
    chunk = np.array([
        [-0.2, 0.1], [0.0, 0.0], [0.1, 0.2], [0.3, 0.3],
        [0.4, 0.1], [0.5, -0.1], [0.7, 0.0], [0.8, 0.2],
    ], dtype=np.float32)
    assert np.array_equal(resample_chunk(chunk, anchor, 1, "tac_fold"), chunk)
    for arm in ("zoh", "spline", "tac_fold"):
        out = resample_chunk(chunk, anchor, 2, arm)
        assert out.shape == chunk.shape
        assert np.allclose(out[1::2], chunk[1::2], atol=1e-5), (arm, out[1::2], chunk[1::2])
    ctac = resample_chunk(chunk, anchor, 2, "ctac_position")
    assert np.allclose(ctac[1::2], chunk[1::2], atol=1e-5)
    assert np.abs(ctac).max() <= 1.0 + 1e-6


def run_eval(args):
    self_check()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_dir = args.policy_dir
    k = args.k
    arms = [a.strip() for a in args.arms.split(",")]
    n_episodes = args.n_episodes
    batch_size = args.batch_size
    start_seed = args.start_seed

    print(f"=== Push-T Multi-Rate Eval (Official Hub Checkpoint) ===")
    print(f"Policy: {policy_dir}")
    print(f"k={k} | episodes={n_episodes} | batch_size={batch_size} | start_seed={start_seed}")
    print(f"Arms: {arms}")
    print(f"Device: {device}\n")

    # Load policy and preprocessors
    policy = DiffusionPolicy.from_pretrained(policy_dir)
    policy.to(device)
    policy.eval()

    from lerobot.processor.converters import (
        batch_to_transition,
        transition_to_batch,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_dir,
        "policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_dir,
        "policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    from safetensors import safe_open
    stats_path = os.path.join(policy_dir, "policy_preprocessor_step_3_normalizer_processor.safetensors")
    with safe_open(stats_path, framework="np") as stats:
        state_min = stats.get_tensor("observation.state.min").astype(np.float32)
        state_max = stats.get_tensor("observation.state.max").astype(np.float32)
        action_min = stats.get_tensor("action.min").astype(np.float32)
        action_max = stats.get_tensor("action.max").astype(np.float32)

    # Make env
    env_cfg = PushtEnv(fps=10, episode_length=300)
    envs = make_env(env_cfg, n_envs=batch_size)
    env = envs['pusht'][0]

    # Empty env pre/post processors
    env_preprocessor = PolicyProcessorPipeline(steps=[])
    env_postprocessor = PolicyProcessorPipeline(steps=[])

    orig_predict = policy.predict_action_chunk
    orig_reset = policy.reset

    results = {
        "policy": "lerobot/diffusion_pusht",
        "k": k,
        "n_episodes": n_episodes,
        "start_seed": start_seed,
        "arms": {}
    }

    for arm in arms:
        print(f"\n--- Running arm: {arm} (k={k}, n={n_episodes}) ---")
        t0 = time.time()

        noise_call = 0

        # Reset the common-random-number stream at each vectorized rollout batch.
        def patched_reset():
            nonlocal noise_call
            noise_call = 0
            return orig_reset()

        policy.reset = patched_reset

        # Monkey-patch policy.predict_action_chunk for this arm.
        def patched_predict(batch, noise=None):
            nonlocal noise_call
            if noise is None:
                generator = torch.Generator(device=device).manual_seed(args.policy_seed + noise_call)
                noise = torch.randn(
                    (batch["observation.state"].shape[0], policy.config.horizon, 2),
                    generator=generator, device=device,
                )
            noise_call += 1
            actions = orig_predict(batch, noise=noise)  # (B, 8, 2)
            if arm == "native" or k == 1:
                return actions
            acts_np = actions.detach().cpu().numpy()
            state_np = batch["observation.state"].detach().cpu().numpy()
            anchors = state_norm_to_action_norm(
                state_np, state_min, state_max, action_min, action_max
            )
            res_np = np.empty_like(acts_np)
            for b in range(acts_np.shape[0]):
                res_np[b] = resample_chunk(acts_np[b], anchors[b], k, arm)
            return torch.from_numpy(res_np).to(actions.device, dtype=actions.dtype)

        policy.predict_action_chunk = patched_predict

        eval_info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            start_seed=start_seed,
            return_episode_data=False,
            max_episodes_rendered=0,
        )

        aggregated = eval_info.get("aggregated", {})
        per_episode = eval_info.get("per_episode", [])
        
        max_rewards = [ep["max_reward"] for ep in per_episode]
        successes = [ep["success"] for ep in per_episode]
        sum_rewards = [ep["sum_reward"] for ep in per_episode]

        cov = np.array(max_rewards)
        succ = np.array(successes)

        arm_res = {
            "avg_max_reward": float(aggregated.get("avg_max_reward", np.mean(cov))),
            "pc_success": float(aggregated.get("pc_success", np.mean(succ) * 100.0)),
            "avg_sum_reward": float(aggregated.get("avg_sum_reward", np.mean(sum_rewards))),
            "n_success": int(np.sum(succ)),
            "n_episodes": len(cov),
            "wall_s": time.time() - t0,
            "max_rewards": [float(x) for x in cov],
            "successes": [bool(x) for x in succ],
        }
        results["arms"][arm] = arm_res

        print(f"[{arm}] Coverage: {arm_res['avg_max_reward']*100:.2f}% | Binary SR: {arm_res['pc_success']:.1f}% ({arm_res['n_success']}/{arm_res['n_episodes']}) | Time: {arm_res['wall_s']:.1f}s")

    # Restore original predict method
    policy.predict_action_chunk = orig_predict
    policy.reset = orig_reset
    env.close()

    out_file = args.out_file or f"result_pusht_hub_k{k}_{args.suffix}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all results to: {out_file}")

    # Summary table
    print("\n" + "="*70)
    print(f"Push-T Multi-Rate Results Summary (k={k}, n={n_episodes}):")
    print(f"{'Arm':<18} | {'Coverage':<12} | {'Binary SR':<14} | {'vs Native':<10}")
    print("-" * 70)
    native_sr = results["arms"].get("native", {}).get("pc_success", 0.0)
    for arm, d in results["arms"].items():
        delta = d["pc_success"] - native_sr
        dl_str = f"{delta:+.1f}pp" if arm != "native" else "—"
        print(f"{arm:<18} | {d['avg_max_reward']*100:6.2f}%     | {d['pc_success']:5.1f}% ({d['n_success']}/{d['n_episodes']}) | {dl_str}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_dir", type=str, default=f"{HERE}/hub_diffusion_pusht")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--arms", type=str, default="native,zoh,spline,tac_fold,ctac_position,bspline_eps_raw")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--policy_seed", type=int, default=20260916)
    parser.add_argument("--suffix", type=str, default="resamplers")
    parser.add_argument("--out_file", type=str, default=None)
    args = parser.parse_args()
    run_eval(args)
