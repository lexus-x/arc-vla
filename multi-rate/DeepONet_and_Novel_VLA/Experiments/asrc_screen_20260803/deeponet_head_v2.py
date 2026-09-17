"""
deeponet_head_v2.py
===================
DeepONet action head v2 — same operator-learning idea, but the BRANCH now reads
the FULL observation token sequence via cross-attention instead of a single
mean-pooled vector. This fixes the information bottleneck that made v1 collapse on
spatially-specific tasks (e.g. "bowl on the stove / cabinet").

Components
----------
* CrossAttnPool: K learned query tokens cross-attend to all prefix tokens
  (Perceiver-style), producing K focused context vectors that preserve spatial
  detail. This IS the DeepONet branch's encoder of the input function.
* Branch: maps the K context vectors -> p branch coefficients.
* Trunk: maps query time tau in [0,1] (with Fourier features) -> p basis values.
* Merge (parameter-free) c (x) phi, then a small output MLP -> action chunk.

Still a DeepONet: a(tau) = OutMLP( branch(obs) (x) trunk(tau) ). Target ~10-15M.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def _mlp(sizes, act=nn.GELU, layernorm=False, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last or last_act:
            if layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


class CrossBlock(nn.Module):
    """Pre-norm cross-attention block: queries attend to context tokens."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))

    def forward(self, q, kv, key_padding_mask):
        qn, kvn = self.ln_q(q), self.ln_kv(kv)
        a, _ = self.attn(qn, kvn, kvn, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        q = q + self.ff(self.ln2(q))
        return q


class CrossAttnPool(nn.Module):
    """K learned queries cross-attend to the full prefix token sequence.

    channel_norm: per-channel normalization ACROSS the token axis (masked mean/var
    over valid tokens, learnable per-channel affine), applied to the raw prefix
    before in_proj. Targets Camera Viewpoints / Sensor Noise, the two categories
    every measured head craters on regardless of operator design. Deliberately NOT
    a per-token LayerNorm: every CrossBlock already LayerNorms kv per-token
    (across channels) via ln_kv, so a second per-token norm is provably redundant
    -- verified empirically (0% change in ctx sensitivity to a synthetic
    per-channel sensor-noise-style perturbation, scratch/check_tokennorm_mechanism.py).
    Normalizing across tokens per channel is the orthogonal axis ln_kv cannot
    reach, and cuts that same sensitivity by ~99% (scratch/check_crosstoken_norm.py).
    """

    def __init__(self, d_in, d_model=512, n_queries=8, n_heads=8, n_blocks=3, channel_norm=False):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.channel_norm = channel_norm
        if channel_norm:
            self.channel_norm_weight = nn.Parameter(torch.ones(d_in))
            self.channel_norm_bias = nn.Parameter(torch.zeros(d_in))
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.n_queries = n_queries
        self.d_model = d_model

    def _channel_norm(self, prefix: Tensor, pad_mask: Tensor, eps: float = 1e-5) -> Tensor:
        # per-channel mean/var across the (valid) token axis -- the axis no
        # existing LayerNorm in this head touches.
        m = pad_mask.float().unsqueeze(-1)                        # (B,N,1)
        n_valid = m.sum(dim=1, keepdim=True).clamp(min=1)          # (B,1,1)
        mean = (prefix * m).sum(dim=1, keepdim=True) / n_valid     # (B,1,d_in)
        var = ((prefix - mean) ** 2 * m).sum(dim=1, keepdim=True) / n_valid
        x = (prefix - mean) / (var + eps).sqrt()
        return x * self.channel_norm_weight + self.channel_norm_bias

    def forward(self, prefix: Tensor, pad_mask: Tensor) -> Tensor:
        # prefix (B,N,d_in); pad_mask (B,N) bool, True=valid token
        B = prefix.shape[0]
        x = self._channel_norm(prefix, pad_mask) if self.channel_norm else prefix
        kv = self.in_proj(x)                              # (B,N,d_model)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B,K,d_model)
        key_padding = ~pad_mask.bool()                    # True = ignore
        for blk in self.blocks:
            q = blk(q, kv, key_padding)
        return q                                          # (B,K,d_model)


class DeepONetHeadV2(nn.Module):
    def __init__(self, context_dim, chunk_size, action_dim,
                 p=256, d_model=512, n_queries=8, n_heads=8, n_blocks=3,
                 branch_hidden=768, trunk_hidden=256, out_hidden=256, n_fourier=16,
                 channel_norm=False, trunk_bandlimit=False, speed_condition=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.p = p
        self.n_fourier = n_fourier
        self.speed_condition = bool(speed_condition)
        self.trunk_bandlimit = trunk_bandlimit

        self.pool = CrossAttnPool(context_dim, d_model, n_queries, n_heads, n_blocks, channel_norm=channel_norm)
        self.branch = _mlp([n_queries * d_model, branch_hidden, p], act=nn.GELU, layernorm=True)
        self.speed_mlp = _mlp([1, branch_hidden, p], act=nn.GELU, layernorm=True) if self.speed_condition else None
        trunk_in = 1 + 2 * n_fourier
        self.trunk = _mlp([trunk_in, trunk_hidden, trunk_hidden, p], act=nn.GELU, layernorm=False)
        self.out_mlp = _mlp([p, out_hidden, action_dim], act=nn.GELU, layernorm=False)
        self.out_bias = nn.Parameter(torch.zeros(action_dim))

        tau = torch.linspace(0.0, 1.0, chunk_size).unsqueeze(-1)  # (T,1)
        self.register_buffer("tau", tau, persistent=False)
        if trunk_bandlimit:
            # Linear spacing: band k contributes k/2 cycles over tau in [0,1], so the top
            # band is n_fourier/2 cycles -- at or below the chunk Nyquist limit
            # (chunk_size/2) whenever n_fourier <= chunk_size. The geometric ladder below
            # puts 2^(k-1) cycles in band k: with n_fourier=16 the top band is 16384 cycles
            # against a chunk of tens of samples, so 10-12 of the 16 bands are ALIASED.
            # Above Nyquist those features alias on the training grid and oscillate rapidly
            # between its samples, compromising off-grid query consistency.
            # Parameter count is IDENTICAL to the geometric variant (trunk_in = 1 + 2*n_fourier
            # either way), so this arm is capacity-matched and isolates aliasing alone.
            # INTEGER cycles: band k = exactly k cycles over tau in [0,1]. Integer cycles are
            # (near-)orthogonal on a uniform grid, so the basis stays full rank with cond ~2,
            # versus cond ~3.7e12 for half-integer (k*pi) spacing which is badly conditioned.
            # Top band = n_fourier cycles = 16 < Nyquist 25 for chunk_size 50.
            freqs = torch.arange(1, n_fourier + 1, dtype=torch.float32) * (2.0 * torch.pi)
        else:
            freqs = (2.0 ** torch.arange(n_fourier)) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)

        # --- off-grid query / alias folding (default OFF; default path untouched) ---
        # Both sign buffers are persistent=False, exactly like tau and freqs: they must
        # NOT appear in state_dict(), or the checkpoint-vs-model shape guard in
        # evaluate_plus.py would see two extra "deeponet" keys and abort every eval.
        self._query_factor = 1
        self._return_query_grid = False
        self._action_interp_factor = 1
        self._folded = False
        self._gaar = False
        self.register_buffer("_sin_sign", torch.ones(n_fourier), persistent=False)
        self.register_buffer("_cos_sign", torch.ones(n_fourier), persistent=False)
        self.register_buffer("_native_tau", tau.clone(), persistent=False)
        self.register_buffer("_gaar_original_freqs", None, persistent=False)

    def set_trunk_fp64(self, on: bool = False):
        """Compute the Fourier carriers in float64, then cast back. Removes the fp32
        phase error on the super-Nyquist octaves (~5e-3 rad at band 15) without
        changing any weight. Free: the carrier matrix is only (T, 1+2F)."""
        self._trunk_fp64 = bool(on)
        return self

    def _fourier(self, tau):
        # tau (T,1) -> (T, 1+2F)
        if getattr(self, '_trunk_fp64', False):
            _t64 = tau.double()
            _a64 = _t64 * self.freqs.double()
            _s64, _c64 = torch.sin(_a64), torch.cos(_a64)
            if self._folded:
                _s64 = _s64 * self._sin_sign.double()
                _c64 = _c64 * self._cos_sign.double()
            return torch.cat([_t64, _s64, _c64], dim=-1).to(tau.dtype)
        ang = tau * self.freqs.to(tau.dtype)              # (T,F)
        if not self._folded:
            return torch.cat([tau, torch.sin(ang), torch.cos(ang)], dim=-1)
        return torch.cat([tau,
                          torch.sin(ang) * self._sin_sign.to(tau.dtype),
                          torch.cos(ang) * self._cos_sign.to(tau.dtype)], dim=-1)

    def _fourier_original(self, tau):
        ang = tau * self._gaar_original_freqs.to(tau.dtype)
        return torch.cat([tau, torch.sin(ang), torch.cos(ang)], dim=-1)

    def _gaar_phi(self, tau):
        """Fold explicit aliases off-grid while returning original features at anchors."""
        native_tau = self._native_tau.to(tau.dtype)
        phi_original = self.trunk(self._fourier_original(native_tau))
        if torch.equal(tau, native_tau):
            return phi_original

        phi_folded_native = self.trunk(self._fourier(native_tau))
        residual = phi_original - phi_folded_native
        pos = tau.squeeze(-1).float().clamp(0.0, 1.0) * (self.chunk_size - 1)
        i0 = pos.floor().long().clamp(0, self.chunk_size - 1)
        i1 = (i0 + 1).clamp(max=self.chunk_size - 1)
        w = (pos - i0.float()).to(residual.dtype).unsqueeze(-1)
        correction = residual[i0] * (1.0 - w) + residual[i1] * w
        phi = self.trunk(self._fourier(tau)) + correction

        # Direct substitution, rather than folded_phi + residual arithmetic, makes
        # exact anchors inherit the checkpoint's original floating-point path.
        matches = tau[:, 0, None].eq(native_tau[:, 0])
        at_anchor = matches.any(dim=1)
        anchor_i = matches.long().argmax(dim=1)
        return torch.where(at_anchor[:, None], phi_original[anchor_i], phi)

    def forward(self, prefix: Tensor, pad_mask: Tensor, speed: Tensor | None = None) -> Tensor:
        ctx = self.pool(prefix, pad_mask)                 # (B,K,d_model)
        c = self.branch(ctx.flatten(1))                   # (B,p)
        if self.speed_mlp is not None:
            if speed is None:
                speed = torch.ones(prefix.shape[0], 1, dtype=prefix.dtype, device=prefix.device)
            c = c + self.speed_mlp(speed.to(prefix.dtype).clamp_min(1e-6).log())
        tau = self.tau.to(prefix.dtype)
        phi = self._phi_eval(tau)
        feat = c.unsqueeze(1) * phi.unsqueeze(0)          # (B,T,p)
        out = self.out_mlp(feat) + self.out_bias          # (B,Tq,A)
        if getattr(self, '_chunk_lowpass', False) and out.shape[1] >= 3:
            _m = out.clone()
            _m[:, 1:-1] = 0.25 * out[:, :-2] + 0.5 * out[:, 1:-1] + 0.25 * out[:, 2:]
            out = _m
        if self._action_interp_factor != 1:
            return self._upsample_actions(out, self._action_interp_factor)
        if out.shape[1] != self.chunk_size and not self._return_query_grid:
            out = self._resample_to_chunk(out)
        return out

    # ---------------- training-free inference smoothing (eval only) ----------------
    def set_supersample(self, k: int = 1):
        """Box-integrate the trunk over each tau cell with k midpoint sub-samples, so
        phi[t] is the CELL AVERAGE rather than a point sample. Content that oscillates
        within a cell -- exactly what the super-Nyquist octave carriers do -- is averaged
        out instead of being sampled at one arbitrary phase. k=1 restores point sampling."""
        self._supersample = max(1, int(k))
        return self

    def set_chunk_lowpass(self, on: bool = False):
        self._chunk_lowpass = bool(on)
        return self

    def _phi_eval(self, tau: Tensor) -> Tensor:
        k = getattr(self, '_supersample', 1)
        if k <= 1:
            return self._gaar_phi(tau) if self._gaar else self.trunk(self._fourier(tau))
        n = tau.shape[0]
        h = 1.0 / max(n - 1, 1)
        off = ((torch.arange(k, device=tau.device, dtype=tau.dtype) + 0.5) / k - 0.5) * h
        ts = (tau.view(-1, 1) + off.view(1, -1)).clamp(0.0, 1.0).reshape(-1, 1)
        ph = self._gaar_phi(ts) if self._gaar else self.trunk(self._fourier(ts))
        return ph.view(n, k, -1).mean(dim=1)

    # ------------------------------------------------------------ off-grid query
    def _resample_to_chunk(self, y: Tensor) -> Tensor:
        """Linear resample (B,Tq,A) on a UNIFORM tau grid back to chunk_size steps."""
        Tq = y.shape[1]
        pos = torch.linspace(0.0, 1.0, self.chunk_size,
                             dtype=y.dtype, device=y.device) * (Tq - 1)
        i0 = pos.floor().long().clamp(0, Tq - 2)
        w = (pos - i0.to(pos.dtype)).view(1, -1, 1)
        return y[:, i0] * (1.0 - w) + y[:, i0 + 1] * w

    @staticmethod
    def _upsample_actions(y: Tensor, factor: int) -> Tensor:
        n = (y.shape[1] - 1) * factor + 1
        pos = torch.linspace(0.0, y.shape[1] - 1, n, dtype=y.dtype, device=y.device)
        i0 = pos.floor().long().clamp(0, y.shape[1] - 1)
        i1 = (i0 + 1).clamp(max=y.shape[1] - 1)
        w = (pos - i0.to(pos.dtype)).view(1, -1, 1)
        return y[:, i0] * (1.0 - w) + y[:, i1] * w

    def set_query_resolution(self, factor: int = 1):
        """Predict on a NESTED denser grid of (chunk_size-1)*factor+1 points, then
        resample back to chunk_size.

        WARNING -- this is a mathematical IDENTITY, not merely a weak probe. This head
        is pointwise in tau (the trunk is a row-wise MLP and the merge is elementwise),
        so every shared grid point is hit with interpolation weight exactly 1.0 and the
        wild intermediate samples are computed and then discarded. Measured relL2 vs
        factor=1 is 0.000000 at factor 2 and 4, for n_fourier 0 and 16, with random AND
        trained weights. Use it as a NULL CONTROL. To actually probe off-grid behaviour
        use set_query_grid(), which is non-nested."""
        factor = int(factor)
        if factor < 1:
            raise ValueError("query factor must be >= 1, got %d" % factor)
        n = (self.chunk_size - 1) * factor + 1
        self.tau = torch.linspace(0.0, 1.0, n, dtype=self.tau.dtype,
                                  device=self.tau.device).unsqueeze(-1)
        self._query_factor = factor
        return self

    def set_query_grid(self, n_points: int):
        """Predict on linspace(0,1,n_points) -- deliberately NOT nested with the
        chunk_size training grid, so the resample genuinely interpolates -- then
        resample back to chunk_size."""
        n_points = int(n_points)
        if n_points < 2:
            raise ValueError("n_points must be >= 2, got %d" % n_points)
        self.tau = torch.linspace(0.0, 1.0, n_points, dtype=self.tau.dtype,
                                  device=self.tau.device).unsqueeze(-1)
        self._query_factor = None
        return self

    def return_query_grid(self):
        """Return every queried action instead of resampling to the training chunk."""
        if self.tau.shape[0] == self.chunk_size:
            raise RuntimeError("return_query_grid() requires a denser query grid")
        self._return_query_grid = True
        return self

    def set_action_interpolation(self, factor: int = 2):
        """Linearly upsample final native-grid actions as a non-operator control."""
        factor = int(factor)
        if factor < 2:
            raise ValueError("action interpolation factor must be >= 2")
        if self.tau.shape[0] != self.chunk_size or self._folded or self._gaar:
            raise RuntimeError("action interpolation requires the unmodified native head")
        self._action_interp_factor = factor
        return self

    def fold_aliases(self, tol: float = 1e-9):
        """Replace every Fourier band by the LOWEST frequency indistinguishable from it
        on the original chunk_size training grid, preserving sin/cos values there.

        With freqs = 2**k * pi, band k is 2**(k-1) CYCLES over tau in [0,1]; for
        chunk_size=50 the Nyquist limit is 24.5 cycles, so 10 of 16 bands are aliased
        (top band = 16384 cycles). On the training grid tau_i = i/(T-1) each
        super-Nyquist band aliases in exact arithmetic to a distinct <=22-cycle sinusoid,
        sometimes with a sign flip that a frequency alone cannot express -- hence the
        per-band _sin_sign / _cos_sign vectors. Off-grid the unfolded bands oscillate
        violently; the folded ones do not, using the SAME trained weights.

        The alias and both signs are DERIVED NUMERICALLY (a search over the only two
        candidates {r, N-r} and both signs, scored by max abs error on the training
        grid), never hardcoded. Everything is verified in float64: the collapse is a
        float64 fact (residual ~1e-11), and float32 physically cannot represent
        32768*pi*tau accurately enough to test a 1e-9 claim. The returned diagnostics
        also report the float32 delta, which is real and non-zero.

        Raises RuntimeError, leaving the module UNMODIFIED, if any band fails tol."""
        N = self.chunk_size - 1
        if N < 2:
            raise RuntimeError("fold_aliases needs chunk_size >= 3, got %d" % self.chunk_size)
        if self._folded:
            raise RuntimeError("fold_aliases() already applied")
        if self.n_fourier == 0:
            self._folded = True
            return {"alias_cycles": [], "sin_sign": [], "cos_sign": [],
                    "max_ongrid_err_f64": 0.0, "max_ongrid_err_f32": 0.0}
        two_pi = 2.0 * math.pi
        t64 = torch.linspace(0.0, 1.0, self.chunk_size, dtype=torch.float64)
        cyc_old = (self.freqs.detach().double() / two_pi).tolist()   # cycles over [0,1]

        alias, s_sin, s_cos, = [], [], []
        for f in cyc_old:
            so, co = torch.sin(t64 * (two_pi * f)), torch.cos(t64 * (two_pi * f))
            r = f % N
            best, closest = None, None
            # Take the LOWEST admissible alias, not merely the lowest-error one. f % N is
            # itself an exact on-grid alias but can sit ABOVE Nyquist (e.g. 1024 -> 44 for
            # N=49): scoring by error alone would happily return that, keep the band
            # super-Nyquist and leave the off-grid pathology completely unrepaired.
            # Exactly one of {r, N-r} is <= N/2, so ascending order guarantees a
            # sub-Nyquist result.
            for a in sorted((r, N - r)):
                sa, ca = torch.sin(t64 * (two_pi * a)), torch.cos(t64 * (two_pi * a))
                for ss in (1.0, -1.0):
                    for cs in (1.0, -1.0):
                        e = max(float((so - ss * sa).abs().max()),
                                float((co - cs * ca).abs().max()))
                        if closest is None or e < closest[0]:
                            closest = (e, a, ss, cs)
                        if e <= tol and best is None:
                            best = (e, a, ss, cs)
                    if best is not None:
                        break
                if best is not None:
                    break
            if best is None:
                raise RuntimeError(
                    "fold_aliases: band at %.6g cycles has no on-grid alias within "
                    "%.1e (closest %.6g cycles, err %.3e)"
                    % (f, tol, closest[1], closest[0]))
            if best[1] > N / 2.0:
                raise RuntimeError(
                    "fold_aliases: alias %.6g cycles for band %.6g is above the grid "
                    "Nyquist limit %.6g -- refusing to claim a repair"
                    % (best[1], f, N / 2.0))
            alias.append(best[1]); s_sin.append(best[2]); s_cos.append(best[3])

        # verify the FULL feature block in float64 BEFORE mutating anything
        def _feat(cyc, ss, cs):
            ang = t64.unsqueeze(-1) * torch.tensor([two_pi * c for c in cyc],
                                                   dtype=torch.float64)
            return torch.cat([t64.unsqueeze(-1),
                              torch.sin(ang) * torch.tensor(ss, dtype=torch.float64),
                              torch.cos(ang) * torch.tensor(cs, dtype=torch.float64)], dim=-1)
        ones = [1.0] * self.n_fourier
        err64 = float((_feat(cyc_old, ones, ones) - _feat(alias, s_sin, s_cos)).abs().max())
        if err64 > tol:
            raise RuntimeError("fold_aliases: on-grid feature mismatch %.3e > %.1e"
                               % (err64, tol))

        dt, dv = self.freqs.dtype, self.freqs.device
        t_native = torch.linspace(0.0, 1.0, self.chunk_size, dtype=dt,
                                  device=dv).unsqueeze(-1)
        f32_before = self._fourier(t_native)
        self.freqs = torch.tensor([two_pi * a for a in alias], dtype=dt, device=dv)
        self._sin_sign = torch.tensor(s_sin, dtype=dt, device=dv)
        self._cos_sign = torch.tensor(s_cos, dtype=dt, device=dv)
        self._folded = True
        err32 = float((f32_before - self._fourier(t_native)).abs().max())
        return {"alias_cycles": alias, "sin_sign": s_sin, "cos_sign": s_cos,
                "max_ongrid_err_f64": err64, "max_ongrid_err_f32": err32}

    def anchor_alias_repair(self, tol: float = 1e-9):
        """Enable Grid-Anchored Alias Repair for frozen-checkpoint evaluation.

        The folded trunk is corrected by linearly interpolating its native-grid
        feature residual. Exact anchors return the original trunk feature directly.
        This is C0 and removes explicit super-Nyquist carriers; it is not band-limited.
        """
        if self.training:
            raise RuntimeError("anchor_alias_repair() is evaluation-only; call eval() first")
        if self._folded or self._gaar:
            raise RuntimeError("anchor_alias_repair() requires an unmodified trunk")
        self._gaar_original_freqs = self.freqs.detach().clone()
        diagnostics = self.fold_aliases(tol=tol)
        self._gaar = True
        return diagnostics

    def num_params(self):
        return sum(q.numel() for q in self.parameters())


if __name__ == "__main__":
    B, N, D, T, A = 2, 600, 960, 50, 32
    head = DeepONetHeadV2(context_dim=D, chunk_size=T, action_dim=A)
    prefix = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool); mask[0, 400:] = False
    y = head(prefix, mask)
    assert y.shape == (B, T, A), y.shape
    print(f"[v2 head] out {tuple(y.shape)}  params={head.num_params()/1e6:.2f}M")
    y.sum().backward(); print("[v2 head] backward OK")

    # channel_norm variant: same shape contract, extra affine params, gradients flow,
    # and masking must exclude padded tokens (row 0 has 200 padded positions) without NaNs
    head_cn = DeepONetHeadV2(context_dim=D, chunk_size=T, action_dim=A, channel_norm=True)
    assert head_cn.pool.channel_norm is True
    assert head_cn.num_params() > head.num_params()
    y_cn = head_cn(prefix, mask)
    assert y_cn.shape == (B, T, A), y_cn.shape
    assert torch.isfinite(y_cn).all(), "NaN/Inf from channel_norm on a padded row"
    y_cn.sum().backward()
    assert head_cn.pool.channel_norm_weight.grad is not None
    print(f"[v2 head channel_norm] out {tuple(y_cn.shape)}  params={head_cn.num_params()/1e6:.2f}M  backward OK")

    # GAAR guard: exact native path plus a finite non-nested off-grid result.
    import copy
    tiny = DeepONetHeadV2(8, 50, 3, p=8, d_model=8, n_queries=2, n_heads=2,
                          n_blocks=1, branch_hidden=8, trunk_hidden=8,
                          out_hidden=8, n_fourier=16).eval()
    gaar = copy.deepcopy(tiny).eval()
    x = torch.randn(1, 7, 8)
    xmask = torch.ones(1, 7, dtype=torch.bool)
    y_native = tiny(x, xmask)
    gaar.anchor_alias_repair()
    assert torch.equal(y_native, gaar(x, xmask)), "GAAR changed the native-grid output"
    gaar.set_query_grid(100)
    y_offgrid = gaar(x, xmask)
    assert y_offgrid.shape == y_native.shape and torch.isfinite(y_offgrid).all()
    print("[v2 head GAAR] native bit-exact; 100-point off-grid path finite")
    gaar.set_query_resolution(2).return_query_grid()
    y_2x = gaar(x, xmask)
    assert y_2x.shape == (1, 99, 3) and torch.isfinite(y_2x).all()
    print("[v2 head GAAR] nested 2x path returns 99 actions")
    linear = copy.deepcopy(tiny).set_action_interpolation(2)
    y_linear = linear(x, xmask)
    assert y_linear.shape == (1, 99, 3)
    assert torch.equal(y_linear[:, ::2], y_native)
    print("[v2 head linear] nested 2x action interpolation preserves all 50 anchors")
