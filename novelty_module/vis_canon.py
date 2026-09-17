"""
vis_canon.py — Training-free test-time visual robustness modules for VLAs.

Track A: IlluminationCanonicalizer — per-image, deterministic canonicalization
  (gray-world white balance -> median-luminance gamma -> percentile contrast
  stretch). Ported from classic illumination-invariance literature (1970s-2000s
  vision); untried in the VLA vault and absent from VLA mainstream (arXiv
  sweeps 2026-09: no test-time illumination canonicalization for VLAs; the only
  camera-robustness module, AnyCamVLA IROS'26, handles viewpoint only).

Track B: input-noise consensus (randomized-smoothing-style action averaging).
  Implemented at the policy level via synchronized clones; no retraining.

Both operate on [0,1] CHW float tensors BEFORE the policy pre-processor, i.e.
in raw image space where illumination perturbations live.
"""

from __future__ import annotations

import torch
from torch import Tensor

IMAGE_KEYS = ("observation.images.image", "observation.images.wrist_image")


def _lum(x: Tensor) -> Tensor:
    """Per-pixel luminance of a CHW [0,1] image -> (1,H,W)."""
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(3, 1, 1)
    return (x * w).sum(dim=0, keepdim=True)


def canonicalize_image(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Deterministic per-image illumination canonicalization.

    x: CHW float in [0,1] (single image, no batch dim).
    Steps (each gain-clamped so mild perturbations are not over-corrected):
      1. Gray-world white balance: remove global color cast.
      2. Median-luminance gamma: map median luma to 0.5 (exposure invariance).
      3. Percentile contrast stretch around 0.5 (2%-98% luminance range).
    """
    orig_dtype = x.dtype
    x = x.float().clamp(0.0, 1.0)

    # 1. gray-world white balance
    means = x.mean(dim=(1, 2))                       # (3,)
    gmean = means.mean()
    scale = (gmean / means.clamp(min=eps)).clamp(0.8, 1.25)
    x = (x * scale.view(3, 1, 1)).clamp(0.0, 1.0)

    # 2. median-luminance gamma
    med = _lum(x).median()
    med = med.clamp(min=eps, max=1.0 - eps)
    gamma = (torch.log(torch.tensor(0.5)) / torch.log(med)).clamp(0.25, 4.0)
    x = x.pow(gamma)

    # 3. percentile contrast stretch, recentered on the median so the median is
    #    mapped exactly to 0.5 regardless of residual gamma shortfall
    lum = _lum(x)
    med_l = lum.median()
    flat = lum.flatten()
    lo = torch.quantile(flat, 0.02)
    hi = torch.quantile(flat, 0.98)
    spread = (hi - lo).clamp(min=eps)
    gain = (0.9 / spread).clamp(0.8, 2.0)
    x = ((x - med_l) * gain + 0.5).clamp(0.0, 1.0)

    return x.to(orig_dtype)


def canonicalize_pin(pin: dict) -> dict:
    """Apply canonicalization to every camera image in a policy-input dict."""
    out = dict(pin)
    for k in IMAGE_KEYS:
        if k in out and torch.is_tensor(out[k]):
            t = out[k]
            # (B,C,H,W) or (C,H,W)
            if t.dim() == 4:
                out[k] = torch.stack([canonicalize_image(t[i]) for i in range(t.shape[0])])
            else:
                out[k] = canonicalize_image(t)
    return out


def noise_pin(pin: dict, sigma: float = 0.02, gen: torch.Generator | None = None) -> dict:
    """Additive Gaussian noise on camera images only (consensus perturbation).

    Noise is drawn on CPU (generator must be a CPU generator) then moved to the
    tensor's device — avoids CUDA-generator restrictions across torch versions.
    """
    out = dict(pin)
    for k in IMAGE_KEYS:
        if k in out and torch.is_tensor(out[k]):
            t = out[k]
            n = torch.randn(t.shape, generator=gen, dtype=torch.float32).to(
                device=t.device, dtype=t.dtype) * sigma
            out[k] = (t + n).clamp(0.0, 1.0)
    return out


def self_test():
    torch.manual_seed(0)
    x = torch.rand(3, 64, 64)
    y = canonicalize_image(x)
    assert y.shape == x.shape and 0.0 <= float(y.min()) and float(y.max()) <= 1.0
    # brightness shift must be largely corrected
    bright = (x * 1.6).clamp(0, 1)
    med_x, med_y = float(_lum(x).median()), float(_lum(canonicalize_image(bright)).median())
    assert abs(med_y - 0.5) < abs(med_x - 0.5) + 0.05, (med_x, med_y)
    # color cast must be reduced
    cast = x * torch.tensor([1.4, 1.0, 0.7]).view(3, 1, 1)
    y2 = canonicalize_image(cast)
    ch = y2.mean(dim=(1, 2))
    assert (ch.max() - ch.min()) < (cast.mean(dim=(1, 2)).max() - cast.mean(dim=(1, 2)).min())
    p = {"observation.images.image": x.unsqueeze(0), "observation.state": torch.zeros(1, 8)}
    q = canonicalize_pin(p)
    assert q["observation.images.image"].shape == (1, 3, 64, 64)
    q2 = noise_pin(p, 0.05)
    assert not torch.equal(q2["observation.images.image"], p["observation.images.image"])
    print(f"[vis_canon] self-test OK (med {med_x:.3f}->{med_y:.3f})")


if __name__ == "__main__":
    self_test()
