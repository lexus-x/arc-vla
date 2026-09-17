"""Paper-format 1X sim-benchmark table (B-spline Policy Table 2a layout) from all result_*.json with k=1."""
import json, glob, re, collections, numpy as np
import json, glob, re, collections, numpy as np
from scipy.stats import binomtest
import sys

rc_mode = len(sys.argv) > 1 and sys.argv[1] == 'rc'
V = collections.defaultdict(dict)   # (pol, task, head, arm) -> seed -> success list
files = glob.glob('result_*.json') + glob.glob('k1_heldout_backup_20260916/result_*.json')
for f in files:
    if 'smoke' in f: continue
    if rc_mode and 'random' in f: continue
    d = json.load(open(f))
    if not isinstance(d, dict) or d.get('k') != 1: continue
    head = 'bspline' if '_bspline' in f else 'blocksum' if '_blocksum' in f else 'step'
    seed = int((re.search(r'_s(\d)', f) or [0, 0])[1]); fold = re.search(r'_f(\d)', f); seed = int(fold[1]) if fold else seed  # folds act as seeds
    for arm, v in d['success'].items():
        if len(v) >= len(V[(d['policy'], d['task'], head, arm)].get(seed, [])): V[(d['policy'], d['task'], head, arm)][seed] = v
def mcn(a, b):
    x = sum(1 for p, q in zip(a, b) if p and not q); y = sum(1 for p, q in zip(a, b) if q and not p)
    return binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
if rc_mode:
    tasks = ['RC-TurnOffSinkFaucet', 'RC-CoffeePressButton', 'RC-TurnOffMicrowave', 'RC-CloseSingleDoor']; hdr = ['Sink faucet', 'Coffee button', 'Microwave', 'Close door', 'Average']
else:
    tasks = ['PushT-v1', 'lift', 'PickCube-v1', 'can', 'square']; hdr = ['PushT', 'Lift', 'PickCube', 'Can', 'Square', 'Average']
ROWS = [
    ('step', 'native', '1X Base'),
    ('bspline', 'native', '1X +B-spline head'),
    ('step', 'bspline_eps05_raw', '1X +B-spline post-hoc (eps .05)'),
    ('step', 'spline', '1X +Spline post-hoc'),
    ('step', 'tac_fold', '1X +TAC-fold post-hoc'),
]
for pol, name in (('dp', 'Diff.'), ('fm', 'FM')):
    print(f'| {name} configuration | ' + ' | '.join(hdr) + ' |'); print('|---' * (len(hdr) + 1) + '|')
    for head, arm, lab in ROWS:
        cs = []
        row_means = []
        row_dls = []
        for t in tasks:
            arm_keys = [arm]
            if arm == 'tac_fold': arm_keys = ['tac_fold', 'tac_fold_satfix']
            elif arm == 'spline': arm_keys = ['spline', 'spline_satfix']
            d = {}
            for ak in arm_keys:
                d = V.get((pol, t, head, ak), {})
                if d: break
            base = V.get((pol, t, 'step', 'native'), {})
            if not d: cs.append('—'); continue
            rates = [100 * np.mean(v) for v in d.values()]; cell = f'{np.mean(rates):.0f}' + (f'±{np.std(rates):.0f}' if len(rates) > 1 else '') + '%'
            row_means.append(np.mean(rates))
            if arm != 'native' or head != 'step':
                pairs = [(d[s], base[s]) for s in d if s in base and len(d[s]) == len(base[s])]
                if pairs:
                    dl = np.mean([100 * (np.mean(a) - np.mean(b)) for a, b in pairs]); ps = [mcn(a, b) for a, b in pairs]
                    cell += f' ({dl:+.0f}{"*" if min(min(ps) * len(ps), 1) < 0.05 else ""})'
                    row_dls.append(dl)
            cs.append(cell)
        if row_means:
            avg_str = f'{np.mean(row_means):.1f}%'
            if row_dls: avg_str += f' ({np.mean(row_dls):+.1f})'
            cs.append(avg_str)
        else:
            cs.append('—')
        print(f'| {name} {lab} | ' + ' | '.join(cs) + ' |')
    print()
print('RoboCasa: 3 held-out folds x 15 episodes, each fold a separately trained model; ' if len(sys.argv) > 1 else '', end=''); print('delta vs same-seed 1X Base, paired exact McNemar (Holm over seeds), * p<0.05. ±sd over 3 seeds where shown; n=100 per seed.')
