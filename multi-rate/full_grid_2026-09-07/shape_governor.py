"""Observation-conditioned governor: learned within-block shape, certified execution.

A small MLP predicts the within-block shape of the executed 8-step chunk from what the
policy already had at chunk start (its 2-frame observation) plus the chunk's own block sums.
The prediction is only an anchor: resample_qp_anchor then enforces exact block sums and
|v|<=1, so however wrong the network is, the executed motion conserves every commanded
total and never exceeds the controller box.

Trained on the policy's own training demos (clipped, as executed), windows cut exactly like
closed-loop chunks. No eval episode is ever touched. Arms (harness.py):
  qp_learned   learned anchor + governor (the method)
  learned_raw  learned anchor, block sums restored, then clipped (ablation: no governor)
  learned_tanh same MLP trained to output the whole chunk through tanh (bounded, no QP) -- the
               reviewer's "why not just bound the network?" alternative; totals not guaranteed
"""
import os
import numpy as np, torch, torch.nn as nn
from resample_math import coarsen_delta
from resample_qp import resample_qp_anchor

T = 8  # executed chunk length (harness executes pred[:8])


def _windows(O, A, k, nd):
    """(2-frame obs, block means) -> zero-sum-per-block residual of the clipped chunk."""
    X, Y = [], []
    for o, a in zip(O, A):
        a = np.clip(np.asarray(a, np.float32)[:, :nd], -1, 1)
        o = np.asarray(o, np.float32)
        a = np.concatenate([a, np.zeros((T, nd), np.float32)])  # post-episode steps = no motion
        for t in range(len(o)):
            c = a[t:t + T]; bs = coarsen_delta(c, k)
            X.append(np.concatenate([o[max(t - 1, 0)], o[t], (bs / k).ravel()]))
            Y.append((c - np.repeat(bs / k, k, 0)).ravel())
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def _net(din, dout):
    return nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, dout))


def fit(O, A, k, nd, steps=4000, seed=0, val_frac=0.1, tanh=False):
    """Fit on demos; report held-out-demo MSE of each anchor vs the true clipped chunk.
    tanh=True: target is the full chunk, output squashed by tanh (learned_tanh arm)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    idx = rng.permutation(len(O)); nv = max(1, int(val_frac * len(O)))
    tr, va = [i for i in idx[nv:]], [i for i in idx[:nv]]
    Xt, Yt = _windows([O[i] for i in tr], [A[i] for i in tr], k, nd)
    if tanh: Yt = Yt + _zoh(Xt, k, nd)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    net = _net(Xt.shape[1], Yt.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    xt, yt = torch.as_tensor((Xt - mu) / sd), torch.as_tensor(Yt)
    for s in range(steps):
        b = torch.randint(0, len(xt), (256,))
        loss = (((torch.tanh(net(xt[b])) if tanh else net(xt[b])) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    model = dict(state=net.state_dict(), mu=mu, sd=sd, k=k, nd=nd, din=Xt.shape[1], dout=Yt.shape[1], tanh=tanh)
    Xv, Yv = _windows([O[i] for i in va], [A[i] for i in va], k, nd)
    with torch.no_grad(): P = net(torch.as_tensor((Xv - mu) / sd)).numpy()
    if tanh: P = np.tanh(P) - _zoh(Xv, k, nd)
    model["val"] = dict(n_windows=len(Xv), mse_zoh=float((Yv ** 2).mean()), mse_learned=float(((P - Yv) ** 2).mean()))
    return model


def _zoh(X, k, nd):  # block means are the last (T//k)*nd inputs -> repeated per step, flat (N, T*nd)
    return np.repeat(X[:, -(T // k) * nd:].reshape(len(X), T // k, nd), k, 1).reshape(len(X), -1)


_CACHE = {}


def _load(path):
    key = (path, os.path.getmtime(path))  # a rewritten file at the same path must not hit a stale entry
    if key not in _CACHE:
        m = torch.load(path, weights_only=False); net = _net(m["din"], m["dout"]); net.load_state_dict(m["state"]); net.eval()
        _CACHE[key] = (m, net)
    return _CACHE[key]


def decode(path, obs2, chunk_nd, governed=True):
    """obs2: (2, obs_dim) raw frames the policy saw; chunk_nd: clipped (8, nd) chunk."""
    m, net = _load(path); k, nd = m["k"], m["nd"]
    if m.get("obs_idx") is not None: obs2 = obs2[:, m["obs_idx"]]  # proprio-only model (harness --gov-obs proprio)
    bs = coarsen_delta(chunk_nd, k)
    x = np.concatenate([obs2[0], obs2[1], (bs / k).ravel()]).astype(np.float32)
    with torch.no_grad(): r = net(torch.as_tensor(((x - m["mu"]) / m["sd"])[None])).numpy()[0].reshape(T, nd)
    if m.get("tanh"): return np.tanh(r).astype(np.float32)  # bounded by construction, totals not enforced
    r -= np.repeat(r.reshape(-1, k, nd).mean(1), k, 0)  # zero-sum per block -> anchor has the exact block sums
    anchor = np.repeat(bs / k, k, 0) + r
    if not governed: return np.clip(anchor, -1, 1).astype(np.float32)
    return resample_qp_anchor(bs, k, anchor_fn=lambda b, kk: anchor)


def fit_bc(O, A, nd, steps=4000, seed=0):
    """mlp_bc arm: same MLP recipe, 2-frame obs -> full clipped 8-step chunk (all dims), run as the
    policy with no DP. Answers "why not just use the small network?". Post-episode padding: zero
    motion on delta dims, last value held on trailing (gripper) dims."""
    torch.manual_seed(seed); X, Y = [], []
    for o, a in zip(O, A):
        a = np.clip(np.asarray(a, np.float32), -1, 1); o = np.asarray(o, np.float32)
        pad = np.concatenate([np.zeros((T, nd), np.float32), np.repeat(a[-1:, nd:], T, 0)], 1)
        a = np.concatenate([a, pad])
        for t in range(len(o)): X.append(np.concatenate([o[max(t - 1, 0)], o[t]])); Y.append(a[t:t + T].ravel())
    X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32); mu, sd = X.mean(0), X.std(0) + 1e-6
    net = _net(X.shape[1], Y.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    x, y = torch.as_tensor((X - mu) / sd), torch.as_tensor(Y)
    for _ in range(steps):
        b = torch.randint(0, len(x), (256,)); loss = ((net(x[b]) - y[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return dict(state=net.state_dict(), mu=mu, sd=sd, din=X.shape[1], dout=Y.shape[1], bc=True)


def decode_bc(path, obs2):
    m, net = _load(path)
    if m.get("obs_idx") is not None: obs2 = obs2[:, m["obs_idx"]]
    x = np.concatenate([obs2[0], obs2[1]]).astype(np.float32)
    with torch.no_grad(): c = net(torch.as_tensor(((x - m["mu"]) / m["sd"])[None])).numpy()[0].reshape(T, -1)
    return np.clip(c, -1, 1).astype(np.float32)


if __name__ == "__main__":  # self-check on synthetic data: anchor learnable, governor exact
    rng = np.random.default_rng(0); nd, k = 3, 4
    ph = [rng.uniform(0, 6.28) + 0.4 * np.arange(30) for _ in range(60)]  # obs (sin, cos of phase) fixes the future
    O = [np.stack([np.sin(p), np.cos(p)], 1).astype(np.float32) for p in ph]
    A = [1.4 * np.sin(p[:, None] + np.arange(nd)).astype(np.float32) for p in ph]  # smooth, saturating
    m = fit(O, A, k, nd, steps=1500)
    assert m["val"]["mse_learned"] < m["val"]["mse_zoh"], m["val"]
    path = "/tmp/_shape_selfcheck.pt"; torch.save(m, path)
    c = np.clip(A[0][:8], -1, 1); out = decode(path, O[0][:2], c)
    assert np.abs(out).max() <= 1 + 1e-6 and np.allclose(coarsen_delta(out, k), coarsen_delta(c, k), atol=1e-4)
    os.remove(path)
    mt = fit(O, A, k, nd, steps=1500, tanh=True); torch.save(mt, path)
    t = decode(path, O[0][:2], c); assert np.abs(t).max() <= 1 and mt["val"]["mse_learned"] < mt["val"]["mse_zoh"], mt["val"]
    os.remove(path)
    torch.save(fit_bc(O, A, nd, steps=500), path); c8 = decode_bc(path, O[0][:2]); assert c8.shape == (8, nd) and np.abs(c8).max() <= 1
    os.remove(path); print("shape_governor self-check ok:", m["val"], "tanh:", mt["val"])
