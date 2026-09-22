"""
Evaluate Push-T at 20 Hz and 40 Hz under TAC-Fold, Cubic Spline, and B-Spline (eps).
Strictly paired across fixed eval seeds.
"""
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from resample_math import resample_tac_fold, resample_spline, resample_zoh
from resample_bspline2 import resample_bspline_eps

def upsample_chunk(chunk_np: np.ndarray, anchor_np: np.ndarray, mult: int, arm: str) -> np.ndarray:
    """Upsample an 8-action position chunk to (8 * mult) positions."""
    deltas = np.diff(np.concatenate([anchor_np[None], chunk_np], axis=0), axis=0) # (8, 2)
    if arm == "zoh":
        fine_deltas = resample_zoh(deltas, mult)
    elif arm == "spline":
        fine_deltas = resample_spline(deltas, mult)
    elif arm == "tac_fold":
        fine_deltas = resample_tac_fold(deltas, mult)
    elif arm == "bspline_eps_raw":
        fine_deltas = resample_bspline_eps(deltas, mult, eps=0.005)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return anchor_np[None] + np.cumsum(fine_deltas, axis=0)

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.envs.factory import make_env
from lerobot.envs.configs import PushtEnv
from lerobot.scripts.lerobot_eval import eval_policy

def state_norm_to_action_norm(state, state_min, state_max, action_min, action_max):
    raw = (state + 1.0) * 0.5 * (state_max - state_min) + state_min
    return 2.0 * (raw - action_min) / (action_max - action_min) - 1.0

def run_eval_rate(fps: int, n_episodes: int = 20, batch_size: int = 10, start_seed: int = 1000, arms=("tac_fold", "spline", "bspline_eps_raw")):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_dir = os.path.join(HERE, "hub_diffusion_pusht")
    mult = fps // 10 # 2 for 20 Hz, 4 for 40 Hz
    
    print(f"\n=======================================================")
    print(f"Running Push-T Evaluation at {fps} Hz (Upsampling {mult}x)")
    print(f"Arms: {arms} | n_episodes: {n_episodes} | batch_size: {batch_size}")
    print(f"=======================================================")

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

    # 30 seconds at given fps
    ep_len = 30 * fps
    env_cfg = PushtEnv(fps=fps, episode_length=ep_len)
    envs = make_env(env_cfg, n_envs=batch_size)
    env = envs['pusht'][0]

    env_preprocessor = PolicyProcessorPipeline(steps=[])
    env_postprocessor = PolicyProcessorPipeline(steps=[])

    orig_predict = policy.predict_action_chunk
    orig_reset = policy.reset

    results = {"fps": fps, "mult": mult, "n_episodes": n_episodes, "arms": {}}

    for arm in arms:
        print(f"\n--- Arm: {arm} ({fps} Hz, {mult}x, n={n_episodes}) ---")
        t0 = time.time()
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
            actions = orig_predict(batch, noise=noise) # (B, 8, 2)
            acts_np = actions.detach().cpu().numpy()
            state_np = batch["observation.state"].detach().cpu().numpy()
            anchors = state_norm_to_action_norm(
                state_np, state_min, state_max, action_min, action_max
            )
            # Upsample to (B, 8 * mult, 2)
            B = acts_np.shape[0]
            res_np = np.empty((B, 8 * mult, 2), dtype=np.float32)
            for b in range(B):
                res_np[b] = upsample_chunk(acts_np[b], anchors[b], mult, arm)
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
        cov = np.array([ep["max_reward"] for ep in per_episode])
        succ = np.array([ep["success"] for ep in per_episode])

        arm_res = {
            "avg_max_reward": float(aggregated.get("avg_max_reward", np.mean(cov))),
            "pc_success": float(aggregated.get("pc_success", np.mean(succ) * 100.0)),
            "n_success": int(np.sum(succ)),
            "n_episodes": len(cov),
            "wall_s": time.time() - t0,
            "max_rewards": [float(x) for x in cov],
            "successes": [bool(x) for x in succ],
        }
        results["arms"][arm] = arm_res
        print(f"[{arm} @ {fps} Hz] Coverage: {arm_res['avg_max_reward']*100:.2f}% | Binary SR: {arm_res['pc_success']:.1f}% ({arm_res['n_success']}/{arm_res['n_episodes']}) | Time: {arm_res['wall_s']:.1f}s")

    policy.predict_action_chunk = orig_predict
    policy.reset = orig_reset
    env.close()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--rates", type=str, default="20,40")
    parser.add_argument("--out_file", type=str, default=os.path.join(HERE, "pusht_20hz_40hz_results.json"))
    args = parser.parse_args()

    rates = [int(r.strip()) for r in args.rates.split(",")]
    master_results = {}

    for fps in rates:
        res = run_eval_rate(fps=fps, n_episodes=args.n_episodes, batch_size=args.batch_size, start_seed=args.start_seed)
        master_results[f"{fps}hz"] = res
        with open(args.out_file, "w") as f:
            json.dump(master_results, f, indent=2)

    print(f"\n=======================================================")
    print(f"All rates {rates} evaluated and saved to {args.out_file}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
