"""
Unified multi-rate evaluation on Push-T (lerobot/diffusion_pusht)
Across 5 key robotics rates:
  - 2.5 Hz (k=4 decimation, VLA latency regime)
  - 5.0 Hz (k=2 decimation, slow-policy regime)
  - 10.0 Hz (native 1x reference)
  - 20.0 Hz (mult=2 upsampling, standard robot controller rate)
  - 50.0 Hz (mult=5 upsampling, teleop / dense execution rate)

Across 5 arms:
  - tac_fold
  - ctac_position (C-TAC)
  - bspline_eps_raw (B-Spline)
  - spline (Cubic Spline)
  - qp_anchor
"""
import argparse, json, os, sys, time
import numpy as np, torch
from scipy.optimize import minimize, LinearConstraint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from resample_math import decimate_and_resample, RESAMPLERS, resample_tac_fold, resample_spline, resample_zoh
from resample_bspline2 import resample_bspline_eps
from resample_qp import resample_qp_anchor

RESAMPLERS['bspline_eps_raw'] = lambda b, k: resample_bspline_eps(b, k, eps=0.005)
RESAMPLERS['qp_anchor'] = lambda b, k: resample_qp_anchor(b, k)

def resample_ctac_position(deltas: np.ndarray, anchor: np.ndarray, k: int) -> np.ndarray:
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
                          constraints=constraints, options={"maxiter": 100, "ftol": 1e-7})
        out[:, dim] = result.x if result.success else x0
    return out.astype(np.float32)

def resample_downsample(chunk_np: np.ndarray, anchor_np: np.ndarray, k: int, arm: str) -> np.ndarray:
    if arm == "native" or k == 1:
        return chunk_np.copy()
    deltas = np.diff(np.concatenate([anchor_np[None], chunk_np], axis=0), axis=0)
    if arm == "ctac_position":
        reconstructed = resample_ctac_position(deltas, anchor_np, k)
    else:
        reconstructed = decimate_and_resample(deltas, k, arm)
    return anchor_np[None] + np.cumsum(reconstructed, axis=0)

def resample_upsample(chunk_np: np.ndarray, anchor_np: np.ndarray, mult: int, arm: str) -> np.ndarray:
    if arm == "native" or mult == 1:
        return chunk_np.copy()
    deltas = np.diff(np.concatenate([anchor_np[None], chunk_np], axis=0), axis=0) # (8, 2)
    if arm == "tac_fold":
        fine_deltas = resample_tac_fold(deltas, mult)
    elif arm == "spline":
        fine_deltas = resample_spline(deltas, mult)
    elif arm == "bspline_eps_raw":
        fine_deltas = resample_bspline_eps(deltas, mult, eps=0.005)
    elif arm == "qp_anchor":
        fine_deltas = resample_qp_anchor(deltas, mult)
    elif arm == "ctac_position":
        fine_deltas = resample_tac_fold(deltas, mult) # fallback anchor for upsampling
    else:
        fine_deltas = resample_zoh(deltas, mult)
    return anchor_np[None] + np.cumsum(fine_deltas, axis=0)

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.envs.factory import make_env
from lerobot.envs.configs import PushtEnv
from lerobot.scripts.lerobot_eval import eval_policy

def state_norm_to_action_norm(state, state_min, state_max, action_min, action_max):
    raw = (state + 1.0) * 0.5 * (state_max - state_min) + state_min
    return 2.0 * (raw - action_min) / (action_max - action_min) - 1.0

def run_single_cell(rate_name, is_upsample, factor, arm, n_episodes, batch_size, start_seed, policy, preprocessor, postprocessor, state_min, state_max, action_min, action_max, device):
    t0 = time.time()
    fps = 10 if not is_upsample else int(10 * factor)
    ep_len = 300 if not is_upsample else int(30 * fps)

    env_cfg = PushtEnv(fps=fps, episode_length=ep_len)
    envs = make_env(env_cfg, n_envs=batch_size)
    env = envs['pusht'][0]

    env_preprocessor = PolicyProcessorPipeline(steps=[])
    env_postprocessor = PolicyProcessorPipeline(steps=[])

    orig_predict = policy.predict_action_chunk
    orig_reset = policy.reset

    noise_call = 0
    def patched_reset():
        nonlocal noise_call
        noise_call = 0
        return orig_reset()
    policy.reset = patched_reset

    def patched_predict(batch, noise=None):
        nonlocal noise_call
        if noise is None:
            generator = torch.Generator(device=device).manual_seed(20260916 + noise_call)
            noise = torch.randn(
                (batch["observation.state"].shape[0], policy.config.horizon, 2),
                generator=generator, device=device,
            )
        noise_call += 1
        actions = orig_predict(batch, noise=noise)
        if arm == "native":
            return actions

        acts_np = actions.detach().cpu().numpy()
        state_np = batch["observation.state"].detach().cpu().numpy()
        anchors = state_norm_to_action_norm(state_np, state_min, state_max, action_min, action_max)

        B = acts_np.shape[0]
        if is_upsample:
            res_np = np.empty((B, 8 * factor, 2), dtype=np.float32)
            for b in range(B):
                res_np[b] = resample_upsample(acts_np[b], anchors[b], factor, arm)
        else:
            res_np = np.empty_like(acts_np)
            for b in range(B):
                res_np[b] = resample_downsample(acts_np[b], anchors[b], factor, arm)
        return torch.from_numpy(res_np).to(actions.device, dtype=actions.dtype)

    policy.predict_action_chunk = patched_predict

    eval_info = eval_policy(
        env=env, policy=policy,
        env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
        preprocessor=preprocessor, postprocessor=postprocessor,
        n_episodes=n_episodes, start_seed=start_seed,
        return_episode_data=False, max_episodes_rendered=0,
    )

    per_episode = eval_info.get("per_episode", [])
    cov = np.array([ep["max_reward"] for ep in per_episode])
    succ = np.array([ep["success"] for ep in per_episode])

    policy.predict_action_chunk = orig_predict
    policy.reset = orig_reset
    env.close()

    res = {
        "avg_max_reward": float(np.mean(cov)),
        "pc_success": float(np.mean(succ) * 100.0),
        "n_success": int(np.sum(succ)),
        "n_episodes": len(cov),
        "wall_s": time.time() - t0,
    }
    print(f"[{rate_name} | {arm}] SR: {res['pc_success']:.1f}% ({res['n_success']}/{res['n_episodes']}) | Cov: {res['avg_max_reward']*100:.2f}% | {res['wall_s']:.1f}s", flush=True)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--out_file", type=str, default=os.path.join(HERE, "pusht_5rates_5arms_results.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_dir = os.path.join(HERE, "hub_diffusion_pusht")
    policy = DiffusionPolicy.from_pretrained(policy_dir)
    policy.to(device)
    policy.eval()

    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_dir, "policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_dir, "policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )

    from safetensors import safe_open
    stats_path = os.path.join(policy_dir, "policy_preprocessor_step_3_normalizer_processor.safetensors")
    with safe_open(stats_path, framework="np") as stats:
        state_min = stats.get_tensor("observation.state.min").astype(np.float32)
        state_max = stats.get_tensor("observation.state.max").astype(np.float32)
        action_min = stats.get_tensor("action.min").astype(np.float32)
        action_max = stats.get_tensor("action.max").astype(np.float32)

    # 5 rates definition: (name, is_upsample, factor)
    rates_cfg = [
        ("2.5Hz", False, 4),  # k=4 decimation
        ("5.0Hz", False, 2),  # k=2 decimation
        ("10.0Hz", False, 1), # native reference
        ("20.0Hz", True, 2),  # 2x upsampling
        ("50.0Hz", True, 5),  # 5x upsampling
    ]

    arms = ["tac_fold", "ctac_position", "bspline_eps_raw", "spline", "qp_anchor"]

    master = {}
    if os.path.exists(args.out_file):
        try:
            master = json.load(open(args.out_file))
        except:
            pass

    for rate_name, is_up, factor in rates_cfg:
        if rate_name not in master:
            master[rate_name] = {}
        for arm in arms:
            if arm in master[rate_name]:
                print(f"Skipping cached {rate_name} | {arm}: {master[rate_name][arm]['pc_success']}%")
                continue
            res = run_single_cell(
                rate_name, is_up, factor, arm,
                args.n_episodes, args.batch_size, args.start_seed,
                policy, preprocessor, postprocessor,
                state_min, state_max, action_min, action_max, device
            )
            master[rate_name][arm] = res
            with open(args.out_file, "w") as f:
                json.dump(master, f, indent=2)

    print("\n" + "="*80)
    print("ALL 5 RATES x 5 ARMS COMPLETED!")
    print(f"Results saved to: {args.out_file}")
    print("="*80)

if __name__ == "__main__":
    main()
