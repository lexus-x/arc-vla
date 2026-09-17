"""Rate-invariance gate. Costs minutes; refuses eval runs that would cost hours.

A rate-integrated head can only be evaluated at an off-native control rate if re-integrating
its ODE at that rate reproduces the SAME trajectory, just sampled more finely. Concretely:
summed pose displacement over a fixed span of physical time must not depend on the
integration rate.

CRITICAL: compare RAW actions, never the normalized chunk. An earlier version of this file
summed the normalized chunk and reported both til and asrc as wildly rate-INCONSISTENT
(ratios 0.03x-7x). That was an artifact: normalization is `(raw - offset)/scale` applied per
step, so summing N steps subtracts `N*offset/scale`, and N doubles from 20 Hz to 40 Hz. In raw
space -- what the env receives, since the postprocessor inverts normalization exactly -- both
variants are rate-invariant to about 1%.

Caveat on interpretation: passing this gate only shows the *commanded* trajectory is
rate-invariant. It says nothing about what the robot does, because LIBERO's OSC controller
saturates. Ground truth for physical motion is logged end-effector position over physical
time (see trace_eef.py), not any action-space quantity.

Exits 0 if every dimension is within --tol of 1.0, else 1, so a launcher can gate on it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--deeponet_head", required=True, choices=["ti", "til", "asrc"])
    parser.add_argument("--fourier", type=int, required=True,
                        help="must match training: 6 for asrc, 16 for til_s0")
    parser.add_argument("--native_rate", type=int, default=20)
    parser.add_argument("--alt_rate", type=int, default=40)
    parser.add_argument("--span_s", type=float, default=1.0,
                        help="physical time over which displacement is compared")
    parser.add_argument("--tol", type=float, default=0.15,
                        help="max allowed |ratio - 1| per dimension")
    parser.add_argument("--task_id", type=int, default=0)
    args = parser.parse_args()

    os.environ["DEEPONET_HEAD"] = args.deeponet_head
    os.environ["DEEPONET_FOURIER"] = str(args.fourier)
    os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"

    import evaluate_multirate_honest as honest
    import evaluate_height_screen as screen
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from rate_integrated_deeponet import RateIntegratedDeepONetHead

    screen.REPLAN = 20
    screen.MAX_STEPS = 440
    screen.CONTROL_FREQ = args.native_rate
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preprocessor, postprocessor) = screen.load_policy("deeponet", args.ckpt, stats)
    heads = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)]
    if len(heads) != 1:
        print(f"[GATE] FATAL: expected 1 rate-integrated head, found {len(heads)}")
        return 1
    head = heads[0]
    offset = head.action_offset[:6].float().cpu().numpy()
    scale = head.action_scale[:6].float().cpu().numpy()
    print(f"[GATE] {args.deeponet_head} head: variant={head.variant} "
          f"n_fourier={head.n_fourier} horizon_s={head.horizon_s}")

    # Build a real observation with the 8-step state history the model requires.
    honest.patch_control_freq(args.native_rate)
    env = screen._make_env(args.task_id)
    env.num_steps_wait = 10
    env.init_state_id = 0
    obs, _ = env.reset(seed=1000)
    for _ in range(10):
        batch = preprocessor(screen._policy_input(obs, env.task_description))
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch)
        obs, _, _, _, _ = env.step(postprocessor(action).to("cpu").float().numpy().reshape(-1))

    batch = preprocessor(screen._policy_input(obs, env.task_description))
    batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}

    # _get_action_chunk needs the history that select_action accumulates, so hook it
    # rather than calling it directly.
    captured: list[np.ndarray] = []
    raw_get_chunk = policy._get_action_chunk

    def capture(*a, **k):
        chunk = raw_get_chunk(*a, **k)
        captured.append(chunk.float().cpu().numpy()[0])
        return chunk

    policy._get_action_chunk = capture

    sums = {}
    for rate in (args.native_rate, args.alt_rate):
        head.set_rate(rate)
        policy.reset()
        captured.clear()
        for _ in range(12):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                policy.select_action(batch)
            if captured:
                break
        if not captured:
            print(f"[GATE] FATAL: no chunk produced at {rate} Hz")
            return 1
        chunk = captured[0]
        n = int(round(args.span_s * rate))
        if chunk.shape[0] < n:
            print(f"[GATE] FATAL: chunk T={chunk.shape[0]} shorter than {n} steps at {rate} Hz")
            return 1
        # MUST convert to RAW action space before summing. The chunk is normalized as
        # (raw - offset)/scale PER STEP, so summing N steps subtracts N*offset/scale -- and N
        # doubles from 20 to 40 Hz. Summing the normalized chunk therefore reports a large
        # spurious difference even when the executed trajectory is identical. The env sees
        # raw, because the postprocessor inverts the normalization exactly.
        raw = chunk[:n, :6] * scale + offset
        sums[rate] = raw.sum(axis=0)
        print(f"[GATE] {rate:3d} Hz | chunk T={chunk.shape[0]:3d} | "
              f"RAW pose sum over {args.span_s}s = {np.round(sums[rate], 5)}")
    env.close()

    native = sums[args.native_rate]
    alt = sums[args.alt_rate]
    # Guard against a near-zero native displacement inflating the ratio.
    scale = np.maximum(np.abs(native), 1e-3)
    ratio = np.abs(alt) / scale
    deviation = np.abs(ratio - 1.0)
    print(f"[GATE] per-dim ratio |{args.alt_rate}Hz|/|{args.native_rate}Hz| = {np.round(ratio, 3)}")
    print(f"[GATE] max deviation from 1.0 = {deviation.max():.3f} (tol {args.tol})")

    if deviation.max() <= args.tol:
        print(f"[GATE] PASS: {args.deeponet_head} COMMANDS a rate-invariant trajectory at "
              f"{args.alt_rate} Hz.")
        print("[GATE] NOTE: this is necessary, not sufficient. Measured end-effector motion "
              "shows a rate-invariant\n       command still under-travels ~2x at 40 Hz, "
              "because LIBERO's OSC consumes actions as\n       per-control-step offsets. "
              "Passing here does NOT predict eval success -- check trace_eef.py.")
        return 0
    print(f"[GATE] FAIL: {args.deeponet_head} does NOT re-integrate consistently at "
          f"{args.alt_rate} Hz. Evaluating it would measure the variant, not multi-rate "
          f"control. Worst dim deviates {deviation.max():.3f} from 1.0.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
