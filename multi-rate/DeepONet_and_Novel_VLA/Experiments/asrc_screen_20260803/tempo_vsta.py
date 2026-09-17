"""TempoVLA-style variable-speed trajectory augmentation for delta actions."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_speed(batch_size: int, low: float, high: float, device: torch.device) -> Tensor:
    if not (0 < low <= 1 <= high):
        raise ValueError("speed range must satisfy 0 < low <= 1 <= high")
    return torch.empty(batch_size, 1, device=device).uniform_(low, high)


def _interp_prefix(prefix: Tensor, positions: Tensor) -> Tensor:
    """Linearly sample cumulative deltas; prefix is (B, source_steps + 1, A)."""
    last = prefix.shape[1] - 1
    lo = positions.floor().long().clamp(0, last)
    hi = (lo + 1).clamp(max=last)
    weight = (positions - lo).unsqueeze(-1)
    return torch.gather(prefix, 1, lo.unsqueeze(-1).expand(-1, -1, prefix.shape[-1])) * (1 - weight) + \
        torch.gather(prefix, 1, hi.unsqueeze(-1).expand(-1, -1, prefix.shape[-1])) * weight


def retime_delta_actions(actions: Tensor, action_is_pad: Tensor, speeds: Tensor,
                         output_steps: int, gripper_index: int) -> tuple[Tensor, Tensor]:
    """Re-time additive actions while preserving accumulated motion and discrete gripper events.

    `actions` must be raw (not normalized) delta actions with at least
    `ceil(max(speed) * output_steps)` source steps. Padded source actions are never
    clamped into a target: affected target steps are marked padded.
    """
    if actions.ndim != 3 or action_is_pad.shape != actions.shape[:2]:
        raise ValueError("actions must be (B,T,A) and action_is_pad must be (B,T)")
    if speeds.shape != (actions.shape[0], 1):
        raise ValueError("speeds must be (B,1)")
    if not 0 <= gripper_index < actions.shape[-1]:
        raise ValueError("invalid gripper_index")

    needed = int(torch.ceil(speeds.max() * output_steps).item())
    if actions.shape[1] < needed:
        raise ValueError("source window is too short for requested speed")

    t = torch.arange(output_steps + 1, device=actions.device, dtype=actions.dtype).view(1, -1)
    positions = t * speeds.to(actions.dtype)
    motion = torch.cat((actions[..., :gripper_index], actions[..., gripper_index + 1:]), dim=-1)
    prefix = torch.cat((torch.zeros_like(motion[:, :1]), motion.cumsum(dim=1)), dim=1)
    retimed_motion = _interp_prefix(prefix, positions).diff(dim=1)

    sample_pos = ((t[:, 1:] - 0.5) * speeds.to(actions.dtype)).round().long()
    sample_pos = sample_pos.clamp(0, actions.shape[1] - 1)
    gripper = torch.gather(actions[..., gripper_index], 1, sample_pos)
    target = torch.cat((retimed_motion[..., :gripper_index], gripper.unsqueeze(-1),
                        retimed_motion[..., gripper_index:]), dim=-1)

    valid_steps = (~action_is_pad).sum(dim=1, keepdim=True).to(actions.dtype)
    target_is_pad = positions[:, 1:] > valid_steps
    return target, target_is_pad


if __name__ == "__main__":
    source = torch.zeros(1, 100, 2)
    source[:, :, 0] = 1.0
    source[:, 50:, 1] = 1.0
    pad = torch.zeros(1, 100, dtype=torch.bool)
    slow, slow_pad = retime_delta_actions(source, pad, torch.tensor([[0.5]]), 50, 1)
    fast, fast_pad = retime_delta_actions(source, pad, torch.tensor([[2.0]]), 50, 1)
    assert torch.allclose(slow[..., 0].sum(), torch.tensor(25.0)) and not slow_pad.any()
    assert torch.allclose(fast[..., 0].sum(), torch.tensor(100.0)) and not fast_pad.any()
