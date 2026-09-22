"""
Rigorously Paired Multi-Rate Evaluation on Push-T comparing:
  1. arc (Ours: Adaptive Rate-optimal Conservative fold)
  2. spline (Cubic Spline baseline)
  3. bspline_eps_raw (B-Spline baseline)

Rates:
  - 2.5 Hz (k=4 decimation)
  - 5.0 Hz (k=2 decimation)
  - 10.0 Hz (native 1x reference)
  - 20.0 Hz (2x upsample)
  - 50.0 Hz (5x upsample)
"""
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from resample_arc import resample_arc
from resample_math import decimate_and_resample, RESAMPLERS, resample_spline, resample_zoh
from resample_bspline2 import resample_bspline_eps

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.envs.factory import make_env
from lerobot.envs.configs import PushtEnv
from lerobot.scripts.lerobot_eval import eval_policy
from safetensors import safe_open

RESAMPLERS['bspline_eps_raw'] = lambda b, k: resample_bspline_eps(b, k, eps=0.005)

def state_norm_to_action_norm(state, state_min, state_max, action_min, action_max):
    raw = (state + 1.0) * 0.5 * (state_max - state_min) + state_min
    return 2.0 * (raw - action_min) / (action_max - action_min) - 1.0

def resample_chunk_down(chunk_np, anchor_np, k, arm):
    if arm == "native" or k == 1:
        return chunk_np.copy()
    deltas = np.diff(np.concatenate([anchor_np[None], chunk_np], axis=0), axis=0)
    reconstructed = decimate_and_resample(deltas, k, arm)
    pos = anchor_np[None] + np.cumsum(reconstructed, axis=0)
    return pos

def resample_chunk_up(chunk_np, anchor_np, mult, arm):
    if arm == "native" or mult == 1:
        return chunk_np.copy()
    deltas = np.diff(np.concatenate([anchor_np[None], chunk_np], axis=0), axis=0)
    if arm == "arc":
        fine_deltas = resample_arc(deltas, mult, is_up=True)
    elif arm == "spline":
        fine_deltas = resample_spline(deltas, mult)
    elif arm == "bspline_eps_raw":
        fine_deltas = resample_bspline_eps(deltas, mult, eps=0.005)
    else:
        fine_deltas = resample_zoh(deltas, mult)
    pos = anchor_np[None] + np.cumsum(fine_deltas, axis=0)
    return pos

def run_rate(rate_name, is_up, factor, arms, n_episodes, batch_size, start_seed, policy_dir, out_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fps = 10 if not is_up else int(10 * factor)
    ep_len = 300 if not is_up else int(30 * fps)

    print(f"\n{'='*70}")
    print(f"Evaluating {rate_name} (fps={fps}, factor={factor}, is_up={is_up})")
    print(f"Arms: {arms} | n={n_episodes} | batch={batch_size}")
    print(f"{'='*70}", flush=True)

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

    stats_path = os.path.join(policy_dir, "policy_preprocessor_step_3_normalizer_processor.safetensors")
    with safe_open(stats_path, framework="np") as stats:
        state_min = stats.get_tensor("observation.state.min").astype(np.float32)
        state_max = stats.get_tensor("observation.state.max").astype(np.float32)
        action_min = stats.get_tensor("action.min").astype(np.float32)
        action_max = stats.get_tensor("action.max").astype(np.float32)

    env_preprocessor = PolicyProcessorPipeline(steps=[])
    env_postprocessor = PolicyProcessorPipeline(steps=[])

    orig_predict = policy.predict_action_chunk
    orig_reset = policy.reset

    rate_results = {}

    for arm in arms:
        t0 = time.time()
        noise_call = 0

        # Deterministic seed locking across arms
        torch.manual_seed(start_seed)
        np.random.seed(start_seed)

        env_cfg = PushtEnv(fps=fps, episode_length=ep_len)
        envs = make_env(env_cfg, n_envs=batch_size)
        env = envs['pusht'][0]

        def patched_reset():
            nonlocal noise_call
            noise_call = 0
            return orig_reset()
        policy.reset = patched_reset

        def patched_predict(batch, noise=None):
            nonlocal noise_call
            if noise is None:
                generator = torch.Generator(device=device).manual_seed(start_seed + noise_call)
                noise = torch.randn(
                    (batch["observation.state"].shape[0], policy.config.horizon, 2),
                    generator=generator, device=device,
                )
            noise_call += 1
            actions = orig_predict(batch, noise=noise)
            if arm == "native" or (not is_up and factor == 1):
                return actions

            acts_np = actions.detach().cpu().numpy()
            state_np = batch["observation.state"].detach().cpu().numpy()
            anchors = state_norm_to_action_norm(state_np, state_min, state_max, action_min, action_max)

            B = acts_np.shape[0]
            if is_up:
                res_np = np.empty((B, 8 * factor, 2), dtype=np.float32)
                for b in range(B):
                    res_np[b] = resample_chunk_up(acts_np[b], anchors[b], factor, arm)
            else:
                res_np = np.empty_like(acts_np)
                for b in range(B):
                    res_np[b] = resample_chunk_down(acts_np[b], anchors[b], factor, arm)
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

        arm_data = {
            "avg_max_reward": float(np.mean(cov)),
            "pc_success": float(np.mean(succ) * 100.0),
            "n_success": int(np.sum(succ)),
            "n_episodes": len(cov),
            "wall_s": time.time() - t0,
            "max_rewards": [float(x) for x in cov],
            "successes": [bool(x) for x in succ],
        }
        rate_results[arm] = arm_data
        print(f"[{rate_name} | {arm:<15}] SR: {arm_data['pc_success']:5.1f}% ({arm_data['n_success']}/{arm_data['n_episodes']}) | Cov: {arm_data['avg_max_reward']*100:6.2f}% | {arm_data['wall_s']:.1f}s", flush=True)

        env.close()

    policy.predict_action_chunk = orig_predict
    policy.reset = orig_reset

    return rate_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--out_file", type=str, default=os.path.join(HERE, "eval_arc_results.json"))
    args = parser.parse_args()

    policy_dir = os.path.join(HERE, "hub_diffusion_pusht")
    arms = ["arc", "spline", "bspline_eps_raw"]

    rates = [
        ("2.5Hz", False, 4),  # k=4 decimation
        ("5.0Hz", False, 2),  # k=2 decimation
        ("10.0Hz", False, 1), # native baseline
        ("20.0Hz", True, 2),  # 2x upsampling
    ]

    all_data = {}
    if os.path.exists(args.out_file):
        try:
            all_data = json.load(open(args.out_file))
        except:
            pass

    for rate_name, is_up, factor in rates:
        res = run_rate(rate_name, is_up, factor, arms, args.n_episodes, args.batch_size, args.start_seed, policy_dir, args.out_file)
        all_data[rate_name] = res
        with open(args.out_file, "w") as f:
            json.dump(all_data, f, indent=2)

    print("\n" + "="*80)
    print("ARC PUSH-T EVALUATION COMPLETED!")
    print(f"Saved to: {args.out_file}")
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
