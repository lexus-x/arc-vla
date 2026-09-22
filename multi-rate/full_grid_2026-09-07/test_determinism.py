import os, sys, time
import numpy as np, torch

HERE = "/home/user/Desktop/multi-rate/full_grid_2026-09-07"
sys.path.insert(0, HERE)

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.envs.factory import make_env
from lerobot.envs.configs import PushtEnv
from lerobot.scripts.lerobot_eval import eval_policy

policy_dir = os.path.join(HERE, "hub_diffusion_pusht")
device = "cuda" if torch.cuda.is_available() else "cpu"

policy = DiffusionPolicy.from_pretrained(policy_dir).to(device)
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

env_preprocessor = PolicyProcessorPipeline(steps=[])
env_postprocessor = PolicyProcessorPipeline(steps=[])

orig_predict = policy.predict_action_chunk
orig_reset = policy.reset

def run_test():
    for run_i in range(2):
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
            return actions

        policy.predict_action_chunk = patched_predict

        env_cfg = PushtEnv(fps=10, episode_length=300)
        envs = make_env(env_cfg, n_envs=10)
        env = envs['pusht'][0]

        info = eval_policy(
            env=env, policy=policy,
            env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor,
            n_episodes=10, start_seed=1000,
            return_episode_data=False, max_episodes_rendered=0,
        )
        env.close()

        per_ep = info.get("per_episode", [])
        succ = [ep["success"] for ep in per_ep]
        rew = [round(ep["max_reward"], 3) for ep in per_ep]
        print(f"Run {run_i}: Success={sum(succ)}/10, Rewards={rew}", flush=True)

if __name__ == "__main__":
    run_test()
