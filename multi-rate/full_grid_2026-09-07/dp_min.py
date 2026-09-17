"""Minimal state-only Diffusion Policy (Chi et al. 2023), pure torch so it runs in BOTH the
`ms3` (ManiSkill, no lerobot/diffusers/einops) and `vla_smolvla_libero` (robosuite) envs.

U-Net vendored verbatim from lerobot/policies/diffusion/modeling_diffusion.py (einops
rearranges replaced by .transpose). Scheduler: DDPM squaredcos_cap_v2, epsilon prediction,
clip_sample -- lerobot/DP defaults. Sampling: DDIM eta=0 (DP's standard fast sampler).
EMA weights are what get evaluated (DP standard). Normalisation: per-dim min-max to [-1,1].

Contract: obs (B, n_obs, obs_dim) -> action chunk (B, horizon, act_dim); execute first
n_action_steps. Defaults n_obs=2, horizon=16, n_action_steps=8, down_dims=(256,512,1024)
= the DP paper's low-dim config.
"""
import math, copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ------------------------------------------------------------------ U-Net (lerobot, verbatim)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, k, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(inp, out, k, padding=k // 2), nn.GroupNorm(n_groups, out), nn.Mish())

    def forward(self, x):
        return self.block(x)


class CondResBlock1d(nn.Module):
    def __init__(self, inp, out, cond_dim, k=5, n_groups=8):
        super().__init__()
        self.out_channels = out
        self.conv1 = Conv1dBlock(inp, out, k, n_groups)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out * 2))  # FiLM scale+bias
        self.conv2 = Conv1dBlock(out, out, k, n_groups)
        self.residual_conv = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, cond):
        out = self.conv1(x)
        ce = self.cond_encoder(cond).unsqueeze(-1)
        out = ce[:, : self.out_channels] * out + ce[:, self.out_channels:]
        out = self.conv2(out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, act_dim, global_cond_dim, down_dims=(256, 512, 1024), k=5, n_groups=8, step_emb=128):
        super().__init__()
        self.step_enc = nn.Sequential(SinusoidalPosEmb(step_emb), nn.Linear(step_emb, step_emb * 4), nn.Mish(),
                                      nn.Linear(step_emb * 4, step_emb))
        cond_dim = step_emb + global_cond_dim
        in_out = [(act_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:]))
        kw = dict(cond_dim=cond_dim, k=k, n_groups=n_groups)
        self.down = nn.ModuleList()
        for i, (di, do) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.down.append(nn.ModuleList([CondResBlock1d(di, do, **kw), CondResBlock1d(do, do, **kw),
                                            nn.Conv1d(do, do, 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([CondResBlock1d(down_dims[-1], down_dims[-1], **kw),
                                  CondResBlock1d(down_dims[-1], down_dims[-1], **kw)])
        self.up = nn.ModuleList()
        for i, (do, di) in enumerate(reversed(in_out[1:])):
            last = i >= len(in_out) - 1
            self.up.append(nn.ModuleList([CondResBlock1d(di * 2, do, **kw), CondResBlock1d(do, do, **kw),
                                          nn.ConvTranspose1d(do, do, 4, 2, 1) if not last else nn.Identity()]))
        self.final = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], k), nn.Conv1d(down_dims[0], act_dim, 1))

    def forward(self, x, t, global_cond):
        x = x.transpose(1, 2)  # b t d -> b d t
        g = torch.cat([self.step_enc(t), global_cond], -1)
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, g), g); skips.append(x); x = down(x)
        for m in self.mid:
            x = m(x, g)
        for r1, r2, up in self.up:
            x = torch.cat((x, skips.pop()), 1); x = up(r2(r1(x, g), g))
        return self.final(x).transpose(1, 2)


# ------------------------------------------------------------------ scheduler + policy
def cosine_alphas_cumprod(T, s=0.008):  # squaredcos_cap_v2 (diffusers)
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = torch.clip(1 - f[1:] / f[:-1], 0, 0.999)
    return torch.cumprod(1 - betas, 0).float()


class MinMax:
    """Per-dim min-max to [-1,1]; constant dims map to 0."""
    def __init__(self, x):
        x = np.asarray(x, np.float32).reshape(-1, x.shape[-1])
        self.lo, self.hi = x.min(0), x.max(0)
        self.rng = np.where(self.hi - self.lo < 1e-8, 1.0, self.hi - self.lo)

    def norm(self, x):  return (np.asarray(x, np.float32) - self.lo) / self.rng * 2 - 1
    def denorm(self, x): return (np.asarray(x, np.float32) + 1) / 2 * self.rng + self.lo
    def state(self):     return {"lo": self.lo, "hi": self.hi, "rng": self.rng}
    @classmethod
    def from_state(cls, s):
        o = cls.__new__(cls); o.lo, o.hi, o.rng = s["lo"], s["hi"], s["rng"]; return o


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_obs=2, horizon=16, n_action_steps=8, T=100, down_dims=(256, 512, 1024)):
        super().__init__()
        self.obs_dim, self.act_dim, self.n_obs, self.horizon, self.n_action_steps, self.T = obs_dim, act_dim, n_obs, horizon, n_action_steps, T
        self.net = ConditionalUnet1D(act_dim, obs_dim * n_obs, down_dims)
        self.register_buffer("acp", cosine_alphas_cumprod(T))

    def loss(self, obs, act):  # obs (B,n_obs,D) act (B,H,A), both already normalised to [-1,1]
        B = act.shape[0]
        t = torch.randint(0, self.T, (B,), device=act.device)
        eps = torch.randn_like(act)
        a = self.acp[t].view(B, 1, 1)
        xt = a.sqrt() * act + (1 - a).sqrt() * eps
        return ((self.net(xt, t, obs.flatten(1)) - eps) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, n_steps=10):  # DDIM eta=0, returns normalised chunk (B,H,A)
        B = obs.shape[0]; g = obs.flatten(1)
        x = torch.randn(B, self.horizon, self.act_dim, device=obs.device)
        ts = torch.linspace(self.T - 1, 0, n_steps).round().long().tolist()
        for i, t in enumerate(ts):
            a_t = self.acp[t]
            eps = self.net(x, torch.full((B,), t, device=obs.device), g)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)  # clip_sample
            if i + 1 < len(ts):
                a_p = self.acp[ts[i + 1]]
                x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps
            else:
                x = x0
        return x


class EMA:
    def __init__(self, model, decay=0.9999, warmup=1000):
        self.m = copy.deepcopy(model).eval(); self.decay, self.warmup, self.n = decay, warmup, 0
        for p in self.m.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # DP's EMA warmup schedule
        for pe, p in zip(self.m.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)


def make_chunks(obs_ep, act_ep, n_obs, horizon):
    """One episode (T,D),(T,A) -> windows: obs (N,n_obs,D) act (N,horizon,A), edge-padded like DP."""
    T = len(act_ep)
    O = np.concatenate([np.repeat(obs_ep[:1], n_obs - 1, 0), obs_ep], 0)
    A = np.concatenate([act_ep, np.repeat(act_ep[-1:], horizon, 0)], 0)
    ob = np.stack([O[t:t + n_obs] for t in range(T)])
    ac = np.stack([A[t:t + horizon] for t in range(T)])
    return ob, ac


def train(policy, obs_chunks, act_chunks, steps=50_000, bs=256, lr=1e-4, dev="cuda", log_every=2000):
    policy.to(dev).train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ema = EMA(policy)
    O = torch.as_tensor(obs_chunks, device=dev); A = torch.as_tensor(act_chunks, device=dev); N = len(O)
    for s in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), device=dev)
        l = policy.loss(O[idx], A[idx])
        opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sched.step(); ema.update(policy)
        if s % log_every == 0 or s == 1:
            print(f"[train] step {s}/{steps} loss={l.item():.5f}", flush=True)
    return ema.m


def prepare_rgb(images, layout="HWC", device=None):
    """Convert (...,H,W,3) / (...,3,H,W) RGB to float (...,3,H,W) in [0,1].

    Float inputs may be [0,1] or raw [0,255]. Never flip axes spatially: the
    RoboCasa data-processing wrapper already matches stored image orientation.
    """
    if isinstance(images, np.ndarray):
        images = np.ascontiguousarray(images)
    x = torch.as_tensor(images, device=device)
    if x.ndim < 3 or layout not in ("HWC", "CHW"):
        raise ValueError("RGB requires at least 3 dimensions and explicit HWC/CHW layout")
    if (x.shape[-1] if layout == "HWC" else x.shape[-3]) != 3:
        raise ValueError("RGB channel dimension must be 3")
    is_uint8 = x.dtype == torch.uint8
    x = x.float()
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 255:
        raise ValueError("RGB values must be finite and in [0,255]")
    if is_uint8 or x.max() > 1:
        x = x / 255.0
    return x.movedim(-1, -3) if layout == "HWC" else x


class VisualDiffusionPolicy(DiffusionPolicy):
    """Additive multimodal DP; one shared RGB encoder, ordered camera embeddings.

    State is already normalized; image inputs remain a separate raw RGB dict.
    ``tiny`` is an explicit CPU-test backbone, never the production default.
    """
    def __init__(self, proprio_dim, act_dim, camera_keys, embedding_dim=64,
                 image_layouts=None, backbone="resnet18", **kwargs):
        camera_keys = tuple(camera_keys)
        if not camera_keys or len(set(camera_keys)) != len(camera_keys):
            raise ValueError("camera_keys must be nonempty and unique")
        super().__init__(proprio_dim + len(camera_keys) * embedding_dim, act_dim, **kwargs)
        self.proprio_dim, self.camera_keys = proprio_dim, camera_keys
        self.image_layouts = dict(image_layouts or {k: "HWC" for k in camera_keys})
        if set(self.image_layouts) != set(camera_keys):
            raise ValueError("image layout keys must match cameras")
        if backbone == "resnet18":
            from torchvision.models import resnet18
            self.encoder = resnet18(weights=None)
            width = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone == "tiny":
            width = 8
            self.encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(1), nn.Flatten())
        else:
            raise ValueError(f"unknown visual backbone: {backbone}")
        self.projection = nn.Linear(width, embedding_dim)

    def encode(self, state, images):
        if state.ndim != 3 or state.shape[1:] != (self.n_obs, self.proprio_dim):
            raise ValueError("state must have shape (B,n_obs,proprio_dim)")
        if set(images) != set(self.camera_keys):
            raise ValueError("image keys must exactly match policy cameras")
        features = [state]
        for key in self.camera_keys:
            x = prepare_rgb(images[key], self.image_layouts[key], state.device)
            if x.ndim != 5 or x.shape[:2] != state.shape[:2]:
                raise ValueError("images must have matching (B,n_obs) leading dimensions")
            z = self.projection(self.encoder(x.flatten(0, 1)))
            features.append(z.reshape(*state.shape[:2], -1))
        return torch.cat(features, dim=-1)

    def loss(self, state, images, act):
        return super().loss(self.encode(state, images), act)

    @torch.no_grad()
    def sample(self, state, images, n_steps=10):
        return super().sample(self.encode(state, images), n_steps=n_steps)


if __name__ == "__main__":  # smoke: shapes, one train step, one sample, EMA copy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = DiffusionPolicy(obs_dim=42, act_dim=8).to(dev)
    o = torch.randn(4, 2, 42, device=dev); a = torch.rand(4, 16, 8, device=dev) * 2 - 1
    assert p.net(a, torch.zeros(4, dtype=torch.long, device=dev), o.flatten(1)).shape == a.shape
    l0 = p.loss(o, a); l0.backward(); assert torch.isfinite(l0)
    s = p.sample(o, n_steps=5); assert s.shape == a.shape and s.abs().max() <= 1.0
    ob, ac = make_chunks(np.zeros((30, 42), np.float32), np.zeros((30, 8), np.float32), 2, 16)
    assert ob.shape == (30, 2, 42) and ac.shape == (30, 16, 8)
    nm = MinMax(np.array([[0., 5.], [2., 5.]])); assert np.allclose(nm.denorm(nm.norm(np.array([[1., 5.]]))), [[1., 5.]])
    print(f"dp_min smoke OK on {dev}; params={sum(x.numel() for x in p.parameters())/1e6:.1f}M")
