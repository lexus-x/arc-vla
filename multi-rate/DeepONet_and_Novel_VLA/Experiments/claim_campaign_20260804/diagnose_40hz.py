"""Localize why every 40 Hz arm scores exactly 0%.

Two independent, cheap tests:

A. OPEN-LOOP ENV SEMANTICS (no policy). Send one fixed delta action for 1.0 s of
   physical time -- 20 steps at 20 Hz, 40 steps at 40 Hz -- and measure end-effector
   displacement. robosuite's OSC_POSE maps action +-1 to a fixed max delta *per control
   step*, so if displacement doubles at 40 Hz the env's velocity scale is rate-dependent
   and a 20 Hz-native action stream necessarily overshoots.

B. HEAD RATE-INVARIANCE. From one fixed observation, integrate the head at 20 Hz and at
   40 Hz and compare summed pose displacement over the same 1.0 s of physical time.
   If native re-integration is correct these must match: that is the whole premise of
   the folding arm. If they do not match, the arm cannot work no matter what the env does.

Run from the campaign dir.
"""

from __future__ import annotations

import os

import numpy as np
import torch

CKPT = ("/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
        "/runs/til_s0/checkpoints/8300")


def test_a_env_semantics():
    print("=" * 70)
    print("TEST A: open-loop env velocity scale (no policy)")
    print("=" * 70)
    import evaluate_multirate_honest as honest
    import evaluate_height_screen as screen

    # Collision-free probes. A -z push saturates against the tabletop (path == net at
    # ~0.05 m), which makes both rates look identical no matter what the controller does.
    for label, axis, magnitude in (("+z lift", 2, 0.3), ("+y lateral", 1, 0.3)):
        action = np.zeros(7, dtype=np.float64)
        action[axis] = magnitude
        action[6] = -1.0
        nets = {}
        for freq in (20, 40):
            honest.patch_control_freq(freq)
            env = screen._make_env(0)
            env.num_steps_wait = int(round(10 * freq / 20))
            env.init_state_id = 0
            env.reset(seed=1000)

            raw = env._env.env._get_observations()
            start = np.array(raw["robot0_eef_pos"], dtype=np.float64)
            n_steps = freq // 2  # 0.5 s of physical time, short of any joint limit
            path = 0.0
            previous = start.copy()
            for _ in range(n_steps):
                raw, _, _, _ = env._env.step(action)
                current = np.array(raw["robot0_eef_pos"], dtype=np.float64)
                path += float(np.linalg.norm(current - previous))
                previous = current
            nets[freq] = float(np.linalg.norm(previous - start))
            print(f"  {label:11s} {freq:2d} Hz | {n_steps:2d} steps = 0.5 s | "
                  f"net {nets[freq]:.4f} m | path {path:.4f} m")
            env.close()
        print(f"  {label:11s} -> 40Hz/20Hz net displacement ratio = "
              f"{nets[40] / max(nets[20], 1e-9):.3f}\n")
    print("  -> ~1.0 means the env delivers the same physical velocity at both rates, so a\n"
          "     20 Hz-native action stream does NOT inherently overshoot at 40 Hz.\n")


def test_b_head_rate_invariance():
    print("=" * 70)
    print("TEST B: head rate-invariance (does re-integration preserve displacement?)")
    print("=" * 70)
    os.environ["DEEPONET_HEAD"] = "til"
    os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"

    import evaluate_multirate_honest as honest
    import evaluate_height_screen as screen
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from rate_integrated_deeponet import RateIntegratedDeepONetHead

    screen.REPLAN = 20
    screen.MAX_STEPS = 440
    screen.CONTROL_FREQ = 20
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preprocessor, postprocessor) = screen.load_policy("deeponet", CKPT, stats)
    head = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)][0]

    # Fill the 8-step state history the same way a rollout would, at 20 Hz.
    honest.patch_control_freq(20)
    env = screen._make_env(0)
    env.num_steps_wait = 10
    env.init_state_id = 0
    obs, _ = env.reset(seed=1000)
    for _ in range(10):
        batch = preprocessor(screen._policy_input(obs, env.task_description))
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch)
        obs, _, _, _, _ = env.step(postprocessor(action).to("cpu").float().numpy().reshape(-1))

    # Same observation, integrated at both rates. Compare 1.0 s of pose displacement.
    # _get_action_chunk cannot be called directly (it needs the 8-step history that
    # select_action accumulates), so hook it and let select_action drive.
    batch = preprocessor(screen._policy_input(obs, env.task_description))
    batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
    captured = []
    raw_get_chunk = policy._get_action_chunk

    def capture(*a, **k):
        chunk = raw_get_chunk(*a, **k)
        captured.append(chunk.float().cpu().numpy()[0])
        return chunk

    policy._get_action_chunk = capture

    sums = {}
    for rate in (20, 40):
        head.set_rate(rate)
        policy.reset()
        captured.clear()
        # Repeatedly feed the SAME observation: history fills with identical states and
        # the first replan produces a chunk conditioned on it. Rate is the only change.
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                policy.select_action(batch)
            if captured:
                break
        arr = captured[0]
        one_second = arr[:rate, :6].sum(axis=0)  # `rate` steps == 1.0 s
        sums[rate] = one_second
        print(f"  {rate:2d} Hz | chunk T={arr.shape[0]:3d} | "
              f"normalized pose sum over 1.0 s = {np.round(one_second, 4)}")

    ratio = np.abs(sums[40]) / np.maximum(np.abs(sums[20]), 1e-8)
    print(f"  ratio |40Hz| / |20Hz| per dim = {np.round(ratio, 3)}")
    print("  -> ~1.0 means re-integration is rate-invariant (folding premise holds).\n"
          "     ~0.5 means the 40 Hz chunk under-travels; ~2.0 means it over-travels.")
    env.close()


if __name__ == "__main__":
    test_a_env_semantics()
    test_b_head_rate_invariance()
