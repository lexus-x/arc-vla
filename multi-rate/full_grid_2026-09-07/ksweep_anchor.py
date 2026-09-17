"""Cheap no-sim MSE gate for resample_qp_anchor, same protocol as ksweep.py (60 real demos,
reconstruction MSE, no sim compute) -- run this BEFORE spending any sim eval budget, exactly
how the original QP was vetted. Separate output file; does not touch ksweep_results.json."""
import sys, json, time, h5py, numpy as np
sys.path.insert(0, '.')
from resample_math import resample_zoh, resample_spline_satfix, resample_tac_fold_satfix, coarsen_delta
from resample_qp import resample_qp, resample_qp_anchor

DATA = {'PushT': 'pusht_rl', 'PickCube': 'pick_rl_joint'}
KS = [2, 3, 4, 5]
methods = {'zoh': resample_zoh, 'spline_satfix': resample_spline_satfix,
           'tac_fold_satfix': resample_tac_fold_satfix, 'qp': resample_qp, 'qp_anchor': resample_qp_anchor}
N_EP = 60
out = {}
for task, h5name in DATA.items():
    h5 = h5py.File(f'/home/user/maniskill_data/{h5name}.h5', 'r')
    meta = json.load(open(f'/home/user/maniskill_data/{h5name}.json'))['episodes']
    for K in KS:
        mse = {m: [] for m in methods}; n = 0
        t0 = time.time()
        for ep in meta[:N_EP]:
            acts = np.clip(np.asarray(h5[f"traj_{int(ep['episode_id'])}"]['actions'], np.float32), -1, 1)
            if len(acts) < 4 * K: continue
            nb = len(acts) // K
            bs = coarsen_delta(acts, K)
            true = acts[:nb * K]
            for name, fn in methods.items():
                r = fn(bs, K)
                mse[name].append(float(np.mean((r[:nb * K] - true) ** 2)))
            n += 1
        row = {m: float(np.mean(v)) for m, v in mse.items()}
        row['n_ep'] = n; row['wall_s'] = time.time() - t0
        out[f'{task}_k{K}'] = row
        print(f"{task} k={K} n={n} | " + " ".join(f"{m}={row[m]:.5f}" for m in methods) + f" | {row['wall_s']:.0f}s", flush=True)
    h5.close()
json.dump(out, open('ksweep_anchor_results.json', 'w'), indent=2)
print("saved ksweep_anchor_results.json")
