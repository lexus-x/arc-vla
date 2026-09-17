"""
Fit a frozen, per-dimension ridge model predicting the within-block k=2 split of a
decimated action chunk, from the SAME information cubic-spline/PCHIP/TAC-fold already
use (the coarse block itself + its two neighboring coarse blocks) -- no live sensor
feedback, no future information beyond what a chunk-resampler already sees.

Train/test split by EPISODE ID so the closed-loop success-rate eval later uses ONLY
episodes never touched during fitting -- avoids overfitting leakage contaminating any
claimed win.
"""
import sys, json
import h5py, numpy as np

sys.path.insert(0, "/home/user/Desktop/multi-rate/vla-vault/scratch")
from sweep_maniskill_decimation_ratios import coarsen_actions  # reuse, read-only import

H5_PATH = "/home/user/maniskill_data/pick_rl_joint.h5"
JSON_PATH = "/home/user/maniskill_data/pick_rl_joint.json"
K = 2
TRAIN_FRAC = 0.6  # leave a large held-out pool for the closed-loop eval

metadata = json.loads(open(JSON_PATH).read())
ep_ids_all = sorted(int(e["episode_id"]) for e in metadata["episodes"])

rng = np.random.RandomState(0)
shuffled = ep_ids_all.copy()
rng.shuffle(shuffled)
n_train = int(TRAIN_FRAC * len(shuffled))
train_ids = set(shuffled[:n_train])
test_ids = sorted(set(shuffled[n_train:]))
print(f"episodes total={len(ep_ids_all)}  train={len(train_ids)}  test(held-out)={len(test_ids)}")

X_by_dim, y_by_dim = None, None
D = None

with h5py.File(H5_PATH, "r") as f:
    for ep_id in train_ids:
        actions = np.asarray(f[f"traj_{ep_id}"]["actions"], dtype=np.float32)
        act_len = (len(actions) // K) * K
        if act_len < K:
            continue
        actions = np.clip(actions[:act_len], -1.0, 1.0)
        delta, gripper = coarsen_actions(actions, K, has_gripper=True)  # (n_blocks, 7)
        n_blocks, d = delta.shape
        if D is None:
            D = d
            X_by_dim = [[] for _ in range(D)]
            y_by_dim = [[] for _ in range(D)]
        raw = actions[:, :-1]  # (act_len, 7) -- the true per-substep motion dims
        for i in range(n_blocks):
            s_im1 = delta[i - 1] if i - 1 >= 0 else np.zeros(D, dtype=np.float32)
            s_i = delta[i]
            s_ip1 = delta[i + 1] if i + 1 < n_blocks else np.zeros(D, dtype=np.float32)
            v1_true = raw[i * K]  # true first sub-step of this block
            for dd in range(D):
                X_by_dim[dd].append([s_im1[dd], s_i[dd], s_ip1[dd], 1.0])
                y_by_dim[dd].append(v1_true[dd])

weights = []
lam = 1.0
for dd in range(D):
    X = np.array(X_by_dim[dd]); y = np.array(y_by_dim[dd])
    A = X.T @ X + lam * np.eye(X.shape[1]); A[-1, -1] -= lam  # no ridge penalty on intercept
    b = X.T @ y
    w = np.linalg.solve(A, b)
    pred = X @ w
    ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2_train = 1 - ss_res / ss_tot
    weights.append(w.tolist())
    print(f"dim {dd}: train R^2={r2_train:.4f}  n={len(y)}  w={np.round(w,4).tolist()}")

out = {
    "k": K, "action_dim_motion": D,
    "weights_per_dim": weights,  # [w_s_im1, w_s_i, w_s_ip1, intercept] per dim
    "train_episode_ids": sorted(train_ids),
    "test_episode_ids_heldout": test_ids,
}
OUT = "/tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad/shape_ridge_weights.json"
with open(OUT, "w") as fo:
    json.dump(out, fo, indent=2)
print(f"\nsaved frozen weights -> {OUT}")
print(f"held-out test episode ids (first 10): {test_ids[:10]} ... total {len(test_ids)}")
