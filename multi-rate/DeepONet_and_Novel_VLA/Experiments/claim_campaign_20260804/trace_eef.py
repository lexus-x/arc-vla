"""Ground truth for the 40 Hz failure: logged end-effector motion over physical time.

Every earlier diagnosis here was inferred from action-space numbers and every one was
confounded:
  * summed NORMALIZED actions -> spurious rate-dependence from accumulated offset
  * summed RAW actions        -> rate-invariant, but says nothing about what the robot does
  * open-loop constant action -> saturated the OSC controller, so all rates looked identical

LIBERO's OSC_POSE is a saturating impedance controller, so physical motion cannot be
predicted from action magnitude. This logs `robot0_eef_pos` every env step during a real
rollout of each arm, puts all arms on a common physical-time grid (step / env_freq), and
reports net displacement and cumulative path length at fixed times.

Reading the table against the reference arm (til_native_20env, which scores 60%):
  ~0.5x reference distance -> arms move at half speed; 11 s is not enough
  ~2.0x                    -> overshoot
  ~1.0x but still failing  -> motion is fine; suspect the state-history window
                              (8 steps = 0.4 s at 20 Hz but only 0.2 s at 40 Hz)
  all three 40 Hz arms identical -> the interventions never reach the env (plumbing bug)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

CHECK_TIMES_S = (0.5, 1.0, 2.0, 5.0)


def trace_arm(arm: str, task_id: int, seed: int, max_time_s: float) -> dict:
    import evaluate_multirate_honest as honest

    model_key, env_freq, head_rate, transform = honest.ARMS[arm]
    backbone, checkpoint, deeponet_head, fourier = honest.MODELS[model_key]

    if backbone == "deeponet":
        os.environ["DEEPONET_HEAD"] = deeponet_head
        os.environ["DEEPONET_FOURIER"] = str(fourier)
        os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"
    else:
        os.environ.pop("DEEPONET_HEAD", None)
        os.environ.pop("DEEPONET_FOURIER", None)
        os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "1"

    import evaluate_height_screen as screen
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    n_steps = int(round(max_time_s * env_freq))
    screen.MAX_STEPS = int(round(honest.WALLCLOCK_S * env_freq))
    screen.REPLAN = int(round(honest.REPLAN_S * env_freq))
    screen.CONTROL_FREQ = head_rate if head_rate is not None else int(honest.NATIVE_HZ)

    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preprocessor, postprocessor) = screen.load_policy(backbone, checkpoint, stats)

    if transform and "spline" in transform.split("+"):
        from rate_integrated_deeponet import RateIntegratedDeepONetHead
        head = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)][0]
        s_off, s_sc = head.action_offset.detach().clone(), head.action_scale.detach().clone()
        target_len = int(np.ceil(honest.HORIZON_S * env_freq))
        raw_get_chunk = policy._get_action_chunk

        def spline_chunk(*a, **k):
            chunk = raw_get_chunk(*a, **k)
            return honest.resample_delta_chunk(chunk, target_len, chunk.shape[1], s_off, s_sc)

        policy._get_action_chunk = spline_chunk

    honest.patch_control_freq(env_freq)
    env = screen._make_env(task_id)
    sim_freq = env._env.env.control_freq
    assert float(sim_freq) == float(env_freq), f"{arm}: sim {sim_freq} != {env_freq}"
    env.num_steps_wait = int(round(honest.SETTLE_STEPS_20HZ * env_freq / honest.NATIVE_HZ))
    env.init_state_id = 0
    obs, _ = env.reset(seed=seed)

    positions = [np.array(env._env.env._get_observations()["robot0_eef_pos"], dtype=np.float64)]
    action_mags = []
    success_step = None
    for step in range(1, n_steps + 1):
        batch = preprocessor(screen._policy_input(obs, env.task_description))
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch)
        raw_action = postprocessor(action).to("cpu").float().numpy().reshape(-1)
        action_mags.append(float(np.linalg.norm(raw_action[:3])))
        obs, _, terminated, _, info = env.step(raw_action)
        positions.append(
            np.array(env._env.env._get_observations()["robot0_eef_pos"], dtype=np.float64))
        if info.get("is_success", False) and success_step is None:
            success_step = step
        if terminated:
            break
    env.close()

    pos = np.array(positions)
    steps_done = len(pos) - 1
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    net = np.linalg.norm(pos - pos[0], axis=1)

    out = {"arm": arm, "env_freq": env_freq, "head_rate": head_rate, "transform": transform,
           "steps": steps_done, "success_step": success_step,
           "mean_cmd_xyz_norm": float(np.mean(action_mags)) if action_mags else None,
           "at": {}}
    for t in CHECK_TIMES_S:
        idx = int(round(t * env_freq))
        if idx <= steps_done:
            out["at"][str(t)] = {"net_m": float(net[idx]), "path_m": float(path[idx])}
    del policy
    torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="til_native_20env,til_naive_40env,"
                                          "til_folding_40env,til_spline_40env")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_time_s", type=float, default=5.0)
    parser.add_argument("--out", default="eef_trace.json")
    args = parser.parse_args()

    results = []
    for arm in args.arms.split(","):
        print(f"\n=== tracing {arm} ===", flush=True)
        results.append(trace_arm(arm, args.task_id, args.seed, args.max_time_s))
        print(f"    {json.dumps(results[-1]['at'])}", flush=True)

    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)

    reference = results[0]
    print(f"\n{'arm':22s} {'env':>4s} {'cmd|xyz|':>9s} " +
          " ".join(f"{'net@' + str(t):>10s}" for t in CHECK_TIMES_S))
    for r in results:
        cells = []
        for t in CHECK_TIMES_S:
            entry = r["at"].get(str(t))
            cells.append(f"{entry['net_m']:10.4f}" if entry else f"{'-':>10s}")
        mag = r["mean_cmd_xyz_norm"]
        print(f"{r['arm']:22s} {r['env_freq']:4d} "
              f"{(f'{mag:9.4f}' if mag is not None else f'{chr(45):>9s}')} " + " ".join(cells))

    print(f"\nratio of net displacement to reference ({reference['arm']}):")
    for r in results[1:]:
        ratios = []
        for t in CHECK_TIMES_S:
            a, b = r["at"].get(str(t)), reference["at"].get(str(t))
            ratios.append(f"{t}s={a['net_m'] / max(b['net_m'], 1e-9):.2f}" if a and b else f"{t}s=-")
        print(f"  {r['arm']:22s} " + "  ".join(ratios))


if __name__ == "__main__":
    main()
