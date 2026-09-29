"""Why the QP exists: offline mechanism test on held-out training demos (no success rates).
For each task, k=4: learned anchor (frozen recipe) -> {clip (learned_raw), QP (qp_learned)}, plus
the reviewer's alternative: same MLP trained to output the full chunk through tanh (bounded, no QP).
Metrics per arm vs the true clipped chunk: MSE, block-total error (lost commanded motion),
end-of-chunk position drift. Stress: noise on the learned residual (network wrong / off-distribution)."""
import sys, os, json, numpy as np, torch
os.chdir('/home/user/Desktop/multi-rate/full_grid_2026-09-07'); sys.path.insert(0, '.')
import shape_governor as sg
from resample_math import coarsen_delta
from resample_qp import resample_qp_anchor
K = 4
TASKS = ['PickCube-v1', 'RollBall-v1', 'PullCube-v1', 'LiftPegUpright-v1', 'PushCube-v1', 'AnymalC-Reach-v1', 'PokeCube-v1', 'StackCube-v1']


def fit_tanh(Xt, Yt_full, steps=4000):
    torch.manual_seed(0); mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    net = sg._net(Xt.shape[1], Yt_full.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    x, y = torch.as_tensor((Xt - mu) / sd), torch.as_tensor(Yt_full)
    for _ in range(steps):
        b = torch.randint(0, len(x), (256,)); loss = ((torch.tanh(net(x[b])) - y[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda X: torch.tanh(net(torch.as_tensor((X - mu) / sd))).detach().numpy()


out = {}
for task in TASKS:
    z = np.load(f'demos_{task}_200.npz', allow_pickle=True); O, A = list(z['O']), list(z['A'])
    nd = A[0].shape[1] - (0 if task == 'AnymalC-Reach-v1' else 1)
    path = f'shape_{task}_k{K}_n200.pt'
    if not os.path.exists(path): torch.save(sg.fit(O, A, K, nd), path)
    m, net = sg._load(path)
    rng = np.random.default_rng(0); idx = rng.permutation(len(O)); nv = max(1, int(.1 * len(O)))
    tr, va = idx[nv:], idx[:nv]
    Xt, Rt = sg._windows([O[i] for i in tr], [A[i] for i in tr], K, nd)
    zoh_t = np.repeat(Xt[:, -(8 // K) * nd:].reshape(-1, 8 // K, nd), K, 1).reshape(len(Xt), -1)
    tanh_pred = fit_tanh(Xt, Rt + zoh_t)
    Xv, Rv = sg._windows([O[i] for i in va], [A[i] for i in va], K, nd)
    bm = Xv[:, -(8 // K) * nd:].reshape(-1, 8 // K, nd)            # block means
    truth = np.repeat(bm, K, 1) + Rv.reshape(-1, 8, nd)           # true clipped chunk
    with torch.no_grad(): P = net(torch.as_tensor((Xv - m['mu']) / m['sd'])).numpy().reshape(-1, 8, nd)
    T = tanh_pred(Xv).reshape(-1, 8, nd)
    res = {}
    for tag, noise in (('clean', 0.0), ('noisy', 0.5)):
        r = P + noise * Rv.std() * np.random.default_rng(1).normal(size=P.shape)
        r -= np.repeat(r.reshape(-1, 8 // K, K, nd).mean(2), K, 1)  # zero-sum per block (as in decode)
        anchor = np.repeat(bm, K, 1) + r
        raw = np.clip(anchor, -1, 1)
        gov = np.stack([resample_qp_anchor(bm[i] * K, K, anchor_fn=lambda b, kk, a=anchor[i]: a) for i in range(len(anchor))])
        arms = {'learned_raw (clip)': raw, 'qp_learned (QP)': gov}
        if noise == 0: arms['tanh net (no QP)'] = T
        for name, v in arms.items():
            bt_err = np.abs(v.reshape(-1, 8 // K, K, nd).sum(2) - bm * K).mean()
            drift = np.abs(v.sum(1) - truth.sum(1)).mean()
            res[f'{tag} | {name}'] = dict(mse=float(((v - truth) ** 2).mean()), block_total_err=float(bt_err), chunk_drift=float(drift))
        res[f'{tag} | anchor_over_limit_frac'] = float((np.abs(anchor) > 1).mean())
    out[task] = res
    print(f'== {task}  (val windows {len(Xv)})')
    for kk, vv in res.items(): print(f'   {kk:32s} {vv if isinstance(vv, float) else "  ".join(f"{a}={b:.4f}" for a, b in vv.items())}')
json.dump(out, open('qp_role_offline_k4.json', 'w'), indent=1)
