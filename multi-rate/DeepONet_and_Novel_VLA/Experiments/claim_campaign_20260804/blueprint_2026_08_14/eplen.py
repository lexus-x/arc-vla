import glob, os, numpy as np, h5py
for f in sorted(glob.glob(os.path.expanduser("~/robomimic_data/*.hdf5"))):
    with h5py.File(f, "r") as h:
        eps = list(h["data"].keys())[:60]
        L = [len(h[f"data/{k}/actions"]) for k in eps]
        D = [np.linalg.norm(np.array(h[f"data/{k}/actions"])[:, :6].sum(0)) for k in eps]
    print(f"{os.path.basename(f).replace('.hdf5',''):12s} median_len={np.median(L):>6.0f} "
          f"median_total_disp={np.median(D):>8.3f}")
