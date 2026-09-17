import sys, json, time, h5py, numpy as np
sys.path.insert(0, '.')
from resample_math import resample_zoh, resample_spline, resample_spline_satfix, resample_tac_fold_satfix, resample_pchip_satfix, coarsen_delta
from resample_qp import resample_qp, resample_hybrid

DATA = {'PushT': 'pusht_rl', 'PickCube': 'pick_rl_joint'}
KS = [2, 3, 4, 5]
methods = {'zoh': resample_zoh, 'spline_satfix': resample_spline_satfix, 'pchip_satfix': resample_pchip_satfix,
           'tac_fold_satfix': resample_tac_fold_satfix, 'qp': resample_qp, 'hybrid': resample_hybrid}
N_EP = 60
out = {}
for task, h5name in DATA.items():
    h5 = h5py.File(f'/home/user/maniskill_data/{h5name}.h5', 'r')
    meta = json.load(open(f'/home/user/maniskill_data/{h5name}.json'))['episodes']
    for K in KS:
        mse = {m: [] for m in methods}; viol = []; n = 0
        t0 = time.time()
        for ep in meta[:N_EP]:
            acts = np.clip(np.asarray(h5[f"traj_{int(ep['episode_id'])}"]['actions'], np.float32), -1, 1)
            if len(acts) < 4 * K: continue
            nb = len(acts) // K
            bs = coarsen_delta(acts, K)
            true = acts[:nb * K]
            raw = resample_spline(bs, K)
            viol.append(float((np.abs(raw[:nb * K]) > 1).any(axis=1).mean()))  # frac of steps where raw spline violates
            for name, fn in methods.items():
                r = fn(bs, K)
                mse[name].append(float(np.mean((r[:nb * K] - true) ** 2)))
            n += 1
        row = {m: float(np.mean(v)) for m, v in mse.items()}
        row['raw_spline_violation_frac'] = float(np.mean(viol)); row['n_ep'] = n; row['wall_s'] = time.time() - t0
        out[f'{task}_k{K}'] = row
        print(f"{task} k={K} n={n} viol={row['raw_spline_violation_frac']:.3f} | " +
              " ".join(f"{m}={row[m]:.5f}" for m in methods) + f" | {row['wall_s']:.0f}s", flush=True)
    h5.close()
json.dump(out, open('ksweep_results.json', 'w'), indent=2)
print("saved ksweep_results.json")
