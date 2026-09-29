"""Dev check (training demos only, no eval episode): does the governor keep its offline advantage when it
sees proprio only (agent.qpos + agent.qvel, the leading state dims) instead of the full state?
Same frozen recipe (shape_governor.fit, seed 0, 10% of training demos held out). Writes dev_proprio_offline.json."""
import json, os, numpy as np, torch, shape_governor as sg
torch.set_num_threads(6)
PROPRIO = {'AnymalC-Reach-v1': 24}  # agent.qpos 12 + agent.qvel 12; Panda tasks: 9 + 9
TASKS = ['PickCube-v1', 'RollBall-v1', 'PullCube-v1', 'LiftPegUpright-v1', 'PushCube-v1', 'AnymalC-Reach-v1', 'PokeCube-v1', 'StackCube-v1']
out = {}
for t in TASKS:
    z = np.load(f'demos_{t}_200.npz', allow_pickle=True); O, A = list(z['O']), list(z['A'])
    nd = A[0].shape[1] - (0 if t == 'AnymalC-Reach-v1' else 1); p = PROPRIO.get(t, 18)
    for k in (4, 2):
        full = sg.fit(O, A, k, nd)['val']; prop = sg.fit([o[:, :p] for o in O], A, k, nd)['val']
        out[f'{t}_k{k}'] = dict(mse_zoh=full['mse_zoh'], mse_full=full['mse_learned'], mse_proprio=prop['mse_learned'], proprio_dims=p)
        r = out[f'{t}_k{k}']; print(f"{t:18s} k={k} zoh={r['mse_zoh']:.4f} full={r['mse_full']:.4f} proprio={r['mse_proprio']:.4f}", flush=True)
json.dump(out, open('dev_proprio_offline.json', 'w'), indent=1)
