"""NCDE-style DeepONet branch for an 8-state VLA history.

This is a PyTorch adaptation, not a reproduction of NCDE-DeepONet: it uses a
fixed-step RK4 solver rather than the reference JAX/Diffrax adaptive Tsit5 path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from deeponet_head_v2 import _mlp


class NCDEStyleDeepONetHead(nn.Module):
    """Causal controlled branch over chronological state-token latents."""

    def __init__(self, context_dim: int, chunk_size: int, action_dim: int, *, p: int = 256,
                 n_fourier: int = 16, history_steps: int = 8, control_dim: int = 8,
                 hidden: int = 840, rk4_substeps: int = 2):
        super().__init__()
        if control_dim < 2 or history_steps < 2 or rk4_substeps < 1:
            raise ValueError("control_dim, history_steps, and rk4_substeps must be valid")
        self.chunk_size, self.action_dim = chunk_size, action_dim
        self.history_steps, self.control_dim = history_steps, control_dim
        self.hidden, self.rk4_substeps = hidden, rk4_substeps
        self.control_proj = nn.Linear(context_dim, control_dim - 1)
        self.init = nn.Linear(control_dim, hidden)
        # Reference NCDE depth; this replaces pool+branch at 0.22% below its current size.
        self.vector_field = _mlp([hidden] + [hidden] * 6 + [hidden * control_dim],
                                 act=nn.GELU, layernorm=True)
        self.read = nn.Linear(hidden, p)
        self.trunk = _mlp([1 + 2 * n_fourier, 256, 256, p], act=nn.GELU, layernorm=False)
        self.out_mlp = _mlp([p, 256, action_dim], act=nn.GELU, layernorm=False)
        self.out_bias = nn.Parameter(torch.zeros(action_dim))
        self.register_buffer("tau", torch.linspace(0.0, 1.0, chunk_size).unsqueeze(-1), persistent=False)
        self.register_buffer("freqs", (2.0 ** torch.arange(n_fourier)) * math.pi, persistent=False)

    def _field(self, z: Tensor) -> Tensor:
        return self.vector_field(z).view(z.shape[0], self.hidden, self.control_dim)

    def _derivative(self, z: Tensor, dx_dt: Tensor) -> Tensor:
        return torch.bmm(self._field(z), dx_dt.unsqueeze(-1)).squeeze(-1)

    def _integrate(self, control: Tensor) -> Tensor:
        """Piecewise-linear control integrated causally with fixed-step RK4."""
        z = torch.tanh(self.init(control[:, 0]))
        dt = 1.0 / (self.history_steps - 1)
        for index in range(self.history_steps - 1):
            dx_dt = (control[:, index + 1] - control[:, index]) / dt
            h = dt / self.rk4_substeps
            for _ in range(self.rk4_substeps):
                k1 = self._derivative(z, dx_dt)
                k2 = self._derivative(z + 0.5 * h * k1, dx_dt)
                k3 = self._derivative(z + 0.5 * h * k2, dx_dt)
                k4 = self._derivative(z + h * k3, dx_dt)
                z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z

    def forward(self, state_latents: Tensor) -> Tensor:
        if state_latents.ndim != 3 or state_latents.shape[1] != self.history_steps:
            raise ValueError(f"ncde_style needs (batch, {self.history_steps}, context_dim), got {tuple(state_latents.shape)}")
        time = torch.linspace(0.0, 1.0, self.history_steps, dtype=state_latents.dtype,
                              device=state_latents.device).view(1, -1, 1)
        control = torch.cat([time.expand(state_latents.shape[0], -1, -1), self.control_proj(state_latents)], dim=-1)
        coefficients = self.read(self._integrate(control))
        tau = self.tau.to(state_latents.dtype)
        angles = tau * self.freqs.to(state_latents.dtype)
        phi = self.trunk(torch.cat([tau, torch.sin(angles), torch.cos(angles)], dim=-1))
        return self.out_mlp(coefficients.unsqueeze(1) * phi.unsqueeze(0)) + self.out_bias

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


if __name__ == "__main__":
    head = NCDEStyleDeepONetHead(960, 50, 32)
    output = head(torch.randn(2, 8, 960))
    assert output.shape == (2, 50, 32)
    assert abs(head.num_params() - 10_366_784) / 10_366_784 < 0.05
    print(f"NCDE_STYLE_OK params={head.num_params()}")
