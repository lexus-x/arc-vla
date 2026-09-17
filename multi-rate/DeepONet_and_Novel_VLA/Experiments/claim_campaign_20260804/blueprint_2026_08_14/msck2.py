import glob, os, sys, numpy as np, h5py
for f in sorted(glob.glob(os.path.expanduser("~/maniskill_data/*.h5"))):
    n = os.path.basename(f).replace(".h5", "")
    with h5py.File(f, "r") as h:
        keys = list(h.keys())
        tr = [k for k in keys if k.startswith("traj_")]
        non = [k for k in keys if not k.startswith("traj_")]
        L, W = [], set()
        for k in tr[:60]:
            a = h[k]["actions"]
            L.append(a.shape[0]); W.add(a.shape[1] if a.ndim == 2 else -1)
    print(f"{n:14s} keys={len(keys):>5} traj={len(tr):>5} non-traj={non[:2]} "
          f"act_dim={sorted(W)} med_len={np.median(L):>6.0f} min={min(L):>5} max={max(L):>5}",
          flush=True)
