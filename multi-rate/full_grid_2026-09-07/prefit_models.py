"""Pre-fit the frozen shape / tanh / bc models for one task, k=4 (same recipe harness.py fits on
demand; pre-fitting only avoids 3 harness mains oversubscribing the CPU with 56 threads each)."""
import sys, os, numpy as np, torch, shape_governor as sg
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 6)))
t = sys.argv[1]; z = np.load(f'demos_{t}_200.npz', allow_pickle=True); O, A = list(z['O']), list(z['A'])
nd = A[0].shape[1] - (0 if t == 'AnymalC-Reach-v1' else 1)  # harness ND = act_dim - n_hold
for p, fn in ((f'shape_{t}_k4_n200.pt', lambda: sg.fit(O, A, 4, nd)), (f'shape_tanh_{t}_k4_n200.pt', lambda: sg.fit(O, A, 4, nd, tanh=True)),
              (f'shape_bc_{t}_n200.pt', lambda: sg.fit_bc(O, A, nd))):
    if not os.path.exists(p): torch.save(fn(), p); print('fit', p, flush=True)
