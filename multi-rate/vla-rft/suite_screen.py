"""Suite-parametrized version of evaluate_height_screen.py's rollout plumbing.

evaluate_height_screen.py (the module train_grpo.py originally imported as
`screen`) hardcodes SUITE="libero_spatial" at module level. Its `load_policy`
and `_policy_input` functions are suite-agnostic (pure policy/observation
plumbing, no suite-specific logic); only SUITE, TASK_IDS, MAX_STEPS,
TRIALS_PER_TASK, DATASET, and `_make_env`'s closure over SUITE are
suite-specific. This module reproduces that same plumbing verbatim
(load_policy, _policy_input copied unchanged) but parametrized by suite name,
so train_grpo.py / rl_eval-style scripts can target Object/Goal/Long the same
way they already target Spatial.

MAX_STEPS per suite matches baseline_eval_suite.py's SUITE_WALLCLOCK values
(already validated against the vault's own working eval): spatial 220,
object 280, goal 300, long 520 (@20Hz). TASK_IDS reuses the same (3, 5)
pinned two-task pilot scope evaluate_height_screen.py uses for Spatial, for
every suite -- this is the existing agreed pilot-scope decision (not a new
choice), just generalized rather than reduced.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

DATASET_BY_SUITE = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_10": "lerobot/libero_10_image",
}
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
TASK_IDS = (3, 5)
TRIALS_PER_TASK = 50
CONTROL_FREQ = 20
REPLAN = 1


def load_policy(head: str, checkpoint: str, dataset_stats):
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from modeling_smolvla_ph import SmolVLAPHPolicy  # noqa: F401 (same class evaluate_height_screen.py uses for head="flow")

    assert head == "flow", "suite_screen only wires the flow (SmolVLA) head, matching this project's scope"
    policy = SmolVLAPHPolicy.from_pretrained(checkpoint, ph_enabled=False)
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


def get_screen(suite: str):
    if suite not in DATASET_BY_SUITE:
        raise ValueError(f"unknown suite {suite!r}, expected one of {list(DATASET_BY_SUITE)}")

    def _make_env(task_id: int):
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv

        bench = benchmark.get_benchmark_dict()[suite]()
        return LiberoEnv(task_suite=bench, task_suite_name=suite, task_id=task_id,
                          obs_type="pixels_agent_pos", control_mode="relative",
                          observation_height=256, observation_width=256)

    return SimpleNamespace(
        SUITE=suite,
        DATASET=DATASET_BY_SUITE[suite],
        TASK_IDS=TASK_IDS,
        MAX_STEPS=MAX_STEPS_BY_SUITE[suite],
        TRIALS_PER_TASK=TRIALS_PER_TASK,
        CONTROL_FREQ=CONTROL_FREQ,
        REPLAN=REPLAN,
        load_policy=load_policy,
        _make_env=_make_env,
        _policy_input=_policy_input,
    )
