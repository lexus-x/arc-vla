"""Minimal state-only Flow Matching policy (conditional/rectified flow matching), sharing the
IDENTICAL U-Net backbone as dp_min.DiffusionPolicy so DP vs FM is an apples-to-apples swap of
training objective + sampler only -- nothing else. train()/EMA/MinMax/make_chunks are imported
from dp_min, not duplicated (same contract: obs (B,n_obs,D) -> action chunk (B,horizon,A)).

Objective: straight-line conditional flow matching (Lipman et al. 2023 / rectified flow).
x0 ~ N(0,1), x1 = action (normalised target), xt = (1-t) x0 + t x1, t ~ U(0,1).
Target velocity is constant along the path: v = x1 - x0. Sampling integrates the learned
velocity field from x0 to x1 with a fixed-step Euler solver (n_steps, default 10 -- matched to
DP's DDIM-10 so wall-clock and effective smoothing are comparable).
"""
import torch
import torch.nn as nn
from dp_min import ConditionalUnet1D, EMA, MinMax, make_chunks, train  # noqa: F401 (re-exported)


class FlowMatchingPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_obs=2, horizon=16, n_action_steps=8, down_dims=(256, 512, 1024)):
        super().__init__()
        self.obs_dim, self.act_dim, self.n_obs, self.horizon, self.n_action_steps = obs_dim, act_dim, n_obs, horizon, n_action_steps
        self.net = ConditionalUnet1D(act_dim, obs_dim * n_obs, down_dims)

    def loss(self, obs, act):  # obs (B,n_obs,D) act (B,H,A), both already normalised to [-1,1]
        B = act.shape[0]
        t = torch.rand(B, device=act.device)
        x0 = torch.randn_like(act)
        xt = (1 - t).view(B, 1, 1) * x0 + t.view(B, 1, 1) * act
        v_target = act - x0
        v_pred = self.net(xt, t * 1000, obs.flatten(1))  # *1000: feed the sinusoidal step-embed a diffusion-like scale
        return ((v_pred - v_target) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, n_steps=10):  # fixed-step Euler ODE, x0~N(0,1) -> x1
        B = obs.shape[0]; g = obs.flatten(1)
        x = torch.randn(B, self.horizon, self.act_dim, device=obs.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=obs.device)
            v = self.net(x, t * 1000, g)
            x = x + v * dt
        return x.clamp(-1, 1)


if __name__ == "__main__":  # smoke: shapes, one train step, one sample
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = FlowMatchingPolicy(obs_dim=42, act_dim=8).to(dev)
    o = torch.randn(4, 2, 42, device=dev); a = torch.rand(4, 16, 8, device=dev) * 2 - 1
    l0 = p.loss(o, a); l0.backward(); assert torch.isfinite(l0)
    s = p.sample(o, n_steps=5); assert s.shape == a.shape and s.abs().max() <= 1.0
    print(f"fm_min smoke OK on {dev}; params={sum(x.numel() for x in p.parameters())/1e6:.1f}M")
