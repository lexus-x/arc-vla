"""Is the cross-rate consistency objective even measuring what control needs?

The asrc checkpoint trained with a live consistency loss (RATE ~5e-4, never 0) yet the
rate-invariance gate fails at every span tested -- worst of all at 0.2 s, the very horizon
the loss constrains. Two very different explanations, with very different fixes:

  (a) WEIGHT TOO SMALL. The objective measures the right thing but at weight 0.1 its share
      of the total loss was ~0.05%, so it never applied pressure. Fix: retrain with a much
      larger --rate_consistency_weight. Costs ~2.2 h per attempt.

  (b) OBJECTIVE MISMATCHED. The loss is already near zero on this checkpoint while the
      displacement the controller actually executes still diverges. Then no weight fixes
      it, because the loss is satisfied by trajectories the gate rejects.

This distinguishes them in minutes: evaluate the head's OWN _consistency() term on a real
observation and print it next to the gate's displacement ratio for the same rates.
Near-zero loss + far-from-1.0 ratio => case (b).
"""

from __future__ import annotations

import os

import numpy as np
import torch

CKPT = "runs/asrc_s0/checkpoints/8300"


def main() -> None:
    os.environ["DEEPONET_HEAD"] = "asrc"
    os.environ["DEEPONET_FOURIER"] = "6"
    os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"

    import evaluate_multirate_honest as honest
    import evaluate_height_screen as screen
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from rate_integrated_deeponet import RateIntegratedDeepONetHead

    screen.REPLAN, screen.MAX_STEPS, screen.CONTROL_FREQ = 20, 440, 20
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preprocessor, postprocessor) = screen.load_policy("deeponet", CKPT, stats)
    head = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)][0]

    # Capture the branch latent c, which is all _raw_rollout needs. A forward hook is
    # required here: assigning over head.branch would be rejected as a submodule.
    captured = {}
    head.branch.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("c", output))

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
    env.close()

    c = captured["c"].float()
    print(f"branch latent c: shape={tuple(c.shape)}")

    with torch.no_grad():
        raw_native = head._raw_rollout(c, 20.0)
        for alt in (25.0, 40.0, 50.0):
            raw_alt = head._raw_rollout(c, alt, head.consistency_horizon_s)
            loss = head._consistency(raw_native, raw_alt, alt)

            # The same 0.2 s displacement the loss compares, in raw units.
            n_native = round(head.consistency_horizon_s * 20)
            n_alt = round(head.consistency_horizon_s * alt)
            d_native = raw_native[:, :n_native, :6].sum(1)[0].cpu().numpy()
            d_alt = raw_alt[:, :n_alt, :6].sum(1)[0].cpu().numpy()
            ratio = np.abs(d_alt) / np.maximum(np.abs(d_native), 1e-8)
            print(f"\n  alt_rate={alt:4.0f} Hz | head's own _consistency loss = {float(loss):.3e}")
            print(f"    raw 0.2s displacement  20 Hz = {np.round(d_native, 5)}")
            print(f"    raw 0.2s displacement {alt:3.0f} Hz = {np.round(d_alt, 5)}")
            print(f"    per-dim ratio = {np.round(ratio, 3)}  "
                  f"max dev from 1.0 = {np.abs(ratio - 1).max():.3f}")

    # Scale reference: how big is the loss relative to the flow-matching loss it competed with?
    print(f"\n  training reference: final mse ~0.098, RATE ~5e-4, weight 0.1")
    print(f"  => consistency share of total loss ~ {0.1 * 5e-4 / 0.098 * 100:.3f}%")


if __name__ == "__main__":
    main()
