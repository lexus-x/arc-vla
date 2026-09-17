"""Time-integrated DeepONet control and the ASRC screening variant.

All heads learn a raw 6D action-velocity proxy and integrate it over time before
returning actions in the policy's normalized space.  ASRC adds band-limited time
features, explicit rate conditioning, and cross-rate trajectory consistency.

``ti`` uses classical RK4.  ``til`` is the controlled VLA adaptation of
TI(L)-DeepONet: a state-only, softmax-normalized four-coefficient MLP weights
the same four RK slopes.  The gripper remains a direct (non-integrated) path.

The six LIBERO pose deltas are integrated component-wise; this is not an SE(3)
twist exponential and must not be described as one.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from deeponet_head_v2 import CrossAttnPool, _mlp


class RateIntegratedDeepONetHead(nn.Module):
    def __init__(self, context_dim, chunk_size, action_dim, *, variant="ti", p=256,
                 d_model=512, n_queries=8, n_heads=8, n_blocks=3,
                 branch_hidden=768, trunk_hidden=256, out_hidden=256,
                 n_fourier=6, consistency_rates=(5, 10, 25, 40, 50),
                 horizon_s=2.5, consistency_horizon_s=0.2):
        super().__init__()
        if variant not in ("ti", "til", "asrc"):
            raise ValueError(f"variant must be ti, til, or asrc, got {variant}")
        if action_dim < 6:
            raise ValueError(f"action_dim must include 6 pose dimensions, got {action_dim}")
        self.variant = variant
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.horizon_s = float(horizon_s)
        self.consistency_horizon_s = float(consistency_horizon_s)
        self.consistency_rates = tuple(int(r) for r in consistency_rates if int(r) != 20)
        self.runtime_rate_hz = 20.0
        self._consistency_loss = None

        self.pool = CrossAttnPool(context_dim, d_model, n_queries, n_heads, n_blocks)
        self.branch = _mlp([n_queries * d_model, branch_hidden, p], layernorm=True)
        self.n_fourier = int(n_fourier) if variant == "asrc" else 0
        safe_cycles = math.floor(min((20, *self.consistency_rates)) * self.horizon_s / 2)
        if self.n_fourier > safe_cycles:
            raise ValueError(f"n_fourier={self.n_fourier} aliases at the minimum trained rate; max={safe_cycles}")
        trunk_in = 1 + 6 + (2 * self.n_fourier) + (1 if variant == "asrc" else 0)
        self.trunk = _mlp([trunk_in, trunk_hidden, trunk_hidden, p])
        self.rhs = _mlp([p, out_hidden, 6])
        self.direct = _mlp([p, out_hidden, action_dim - 6])
        self.learnable_rk4 = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4)
        ) if variant == "til" else None
        freqs = torch.arange(1, self.n_fourier + 1, dtype=torch.float32) * (2 * math.pi)
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("action_offset", torch.zeros(action_dim), persistent=False)
        self.register_buffer("action_scale", torch.ones(action_dim), persistent=False)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def configure_action_stats(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.action_offset.dtype,
                               device=self.action_offset.device).flatten()
        std = torch.as_tensor(std, dtype=self.action_scale.dtype,
                              device=self.action_scale.device).flatten()
        if mean.numel() != std.numel() or mean.numel() < 6 or mean.numel() > self.action_dim:
            raise ValueError(f"invalid action mean/std shapes: {tuple(mean.shape)}, {tuple(std.shape)}")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("action mean/std must be finite and std must be positive")
        self.action_offset.zero_(); self.action_scale.fill_(1.0)
        self.action_offset[:mean.numel()].copy_(mean)
        self.action_scale[:std.numel()].copy_(std)
        return self

    def set_rate(self, rate_hz):
        rate_hz = float(rate_hz)
        if rate_hz <= 0 or round(self.horizon_s * rate_hz) < 1:
            raise ValueError(f"invalid control rate {rate_hz}")
        self.runtime_rate_hz = rate_hz
        return self

    def _feature(self, c: Tensor, pose: Tensor, time_s: Tensor, rate_hz: float):
        tau = time_s / self.horizon_s
        parts = [tau]
        if self.n_fourier:
            ang = tau * self.freqs.to(tau.dtype)
            parts.extend((torch.sin(ang), torch.cos(ang)))
        parts.append(pose)
        if self.variant == "asrc":
            parts.append(torch.full_like(tau, math.log2(rate_hz / 20.0)))
        phi = self.trunk(torch.cat(parts, dim=-1))
        return c * phi

    def _raw_rollout(self, c: Tensor, rate_hz: float, horizon_s=None):
        n_steps = math.ceil((self.horizon_s if horizon_s is None else horizon_s) * rate_hz)
        dt = 1.0 / rate_hz
        pose = c.new_zeros(c.shape[0], 6)
        rows = []
        for i in range(n_steps):
            t0 = c.new_full((c.shape[0], 1), i * dt)
            k1 = self.rhs(self._feature(c, pose, t0, rate_hz))
            k2 = self.rhs(self._feature(c, pose + 0.5 * dt * k1, t0 + 0.5 * dt, rate_hz))
            k3 = self.rhs(self._feature(c, pose + 0.5 * dt * k2, t0 + 0.5 * dt, rate_hz))
            k4 = self.rhs(self._feature(c, pose + dt * k3, t0 + dt, rate_hz))
            if self.learnable_rk4 is None:
                delta = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                alpha = torch.softmax(self.learnable_rk4(pose), dim=-1)
                delta = dt * sum(alpha[:, i:i + 1] * k for i, k in enumerate((k1, k2, k3, k4)))
            mid_pose = pose + 0.5 * delta
            direct = self.direct(self._feature(c, mid_pose, t0 + 0.5 * dt, rate_hz))
            rows.append(torch.cat((delta, direct), dim=-1))
            pose = pose + delta
        return torch.stack(rows, dim=1)

    def _normalize(self, raw: Tensor):
        return (raw - self.action_offset.to(raw.dtype)) / self.action_scale.to(raw.dtype)

    def _consistency(self, raw_native: Tensor, raw_alt: Tensor, alt_rate: float):
        def endpoint(raw, rate):
            n = round(self.consistency_horizon_s * rate)
            return raw[:, :n, :6].sum(dim=1)

        scale = self.action_scale[:6].to(raw_native.dtype).clamp_min(1e-6)
        loss = F.mse_loss(endpoint(raw_native, 20) / scale,
                          endpoint(raw_alt, alt_rate) / scale)
        if self.action_dim > 6:
            gscale = self.action_scale[6].to(raw_native.dtype).clamp_min(1e-6)
            native_i = round(self.consistency_horizon_s * 20) - 1
            alt_i = round(self.consistency_horizon_s * alt_rate) - 1
            loss = loss + 0.1 * F.mse_loss(raw_native[:, native_i, 6] / gscale,
                                           raw_alt[:, alt_i, 6] / gscale)
        return loss

    def forward(self, prefix: Tensor, pad_mask: Tensor) -> Tensor:
        ctx = self.pool(prefix, pad_mask)
        c = self.branch(ctx.flatten(1))
        rate = 20.0 if self.training else self.runtime_rate_hz
        raw = self._raw_rollout(c, rate)
        self._consistency_loss = raw.new_zeros(())
        if self.training and self.variant == "asrc" and self.consistency_rates:
            idx = int(torch.randint(len(self.consistency_rates), (), device=raw.device))
            alt_rate = self.consistency_rates[idx]
            raw_alt = self._raw_rollout(c, alt_rate, self.consistency_horizon_s)
            self._consistency_loss = self._consistency(raw, raw_alt, alt_rate)
        return self._normalize(raw)

    def take_consistency_loss(self):
        loss, self._consistency_loss = self._consistency_loss, None
        return loss


def _self_check():
    torch.manual_seed(0)
    head = RateIntegratedDeepONetHead(8, 50, 7, variant="asrc", p=8, d_model=8,
                                     n_queries=2, n_heads=2, n_blocks=1,
                                     branch_hidden=16, trunk_hidden=16, out_hidden=16,
                                     n_fourier=2)
    head.configure_action_stats([0.1] * 7, [2.0] * 7).eval()
    for p in head.rhs.parameters():
        p.data.zero_()
    head.rhs[-1].bias.data.copy_(torch.tensor([1., 2., 3., 4., 5., 6.]))
    prefix, mask = torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool)
    totals = []
    for rate in (5, 20, 50):
        y = head.set_rate(rate)(prefix, mask)
        assert y.shape == (2, math.ceil(2.5 * rate), 7)
        raw_pose = y[:, :, :6] * 2.0 + 0.1
        totals.append(raw_pose[:, :round(2.0 * rate)].sum(dim=1))
    assert torch.allclose(totals[0], totals[1], atol=2e-5)
    assert torch.allclose(totals[1], totals[2], atol=2e-5)
    head.train()(prefix, mask)
    assert torch.isfinite(head.take_consistency_loss())
    til = RateIntegratedDeepONetHead(8, 50, 7, variant="til", p=8, d_model=8,
                                     n_queries=2, n_heads=2, n_blocks=1,
                                     branch_hidden=16, trunk_hidden=16, out_hidden=16)
    alpha = torch.softmax(til.learnable_rk4(torch.randn(3, 6)), dim=-1)
    assert torch.all(alpha > 0) and torch.allclose(alpha.sum(-1), torch.ones(3))
    assert torch.isfinite(til(prefix, mask)).all()
    try:
        RateIntegratedDeepONetHead(8, 50, 7, variant="asrc", n_fourier=7)
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe 5 Hz Fourier basis was accepted")
    print("rate-integrated head self-check passed")


if __name__ == "__main__":
    _self_check()
