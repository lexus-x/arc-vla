"""Paper-format table (B-spline Policy Table 2 layout) from step-head result_*.json."""
import json, glob, collections, sys
from scipy.stats import binomtest
S = collections.defaultdict(dict)  # (policy,k,task) -> arm -> success list (largest n wins per arm)
files = collections.defaultdict(dict)  # (policy,k,task) -> arm -> source filename
conflicts = []
for f in sorted(glob.glob('result_*.json')):
    with open(f) as fh:
        d = json.load(fh)
    if d.get('n', 0) < 10 or d.get('head', 'step') != 'step':
        continue
    if any(tag in f for tag in ['_blocksum', '_bspline', '_noise', '_smoke', '_s0_', '_s1_', '_s2_', '_f0_', '_f1_', '_f2_', '_rc4', '_100k', '_cache', '_qpeps', 'gripper_sync', '_test']):
        continue
    key = (d['policy'], d['k'], d['task'])
    for a, v in d['success'].items():
        existing = S[key].get(a)
        if existing is not None and len(existing) == len(v):
            rate_a = 100 * sum(existing) / len(existing)
            rate_b = 100 * sum(v) / len(v)
            if abs(rate_a - rate_b) > 10:
                conflicts.append((key, a, files[key][a], rate_a, f, rate_b))
        if len(v) > len(S[key].get(a, [])):
            S[key][a] = v
            files[key][a] = f
if conflicts:
    for conflict in conflicts:
        print(conflict, file=sys.stderr)
    sys.exit(1)
tasks = ['PushT-v1', 'lift', 'PickCube-v1', 'can', 'square']; hdr = ['PushT', 'Lift', 'PickCube', 'Can', 'Square']
def rate(pol, k, t, a):
    v = S.get((pol, k, t), {}).get(a); return None if v is None else 100 * sum(v) / len(v)
def cell(pol, k, t, a, ref='native'):
    r = rate(pol, k, t, a)
    if r is None: return '—'
    v = S[(pol, k, t)][a]; w = S[(pol, k, t)].get(ref) or S.get((pol, 2, t), {}).get(ref)
    if w is None or len(v) != len(w): return f'{r:.0f}%'
    b = sum(1 for x, y in zip(v, w) if x and not y); c = sum(1 for x, y in zip(v, w) if y and not x)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return f'{r:.0f}% ({r - 100 * sum(w) / len(w):+.0f}{"*" if p < 0.05 else ""})'
ROWS = [('1X Base', 2, 'native', None),
        ('1X +B-spline post-hoc eps.005', 1, 'bspline_eps_raw', 'native'), ('1X +B-spline post-hoc eps.05', 1, 'bspline_eps05_raw', 'native'),
        ('1X +QP-eps.05 (ours)', 1, 'qp_eps05', 'native'), ('1X +QP-eps.10 (ours)', 1, 'qp_eps10', 'native'),
        ('2X naive (ZOH)', 2, 'zoh', 'native'), ('2X +spline+satfix', 2, 'spline_satfix', 'native'),
        ('2X +TAC-Fold+satfix', 2, 'tac_fold_satfix', 'native'), ('2X +QP (ours)', 2, 'qp', 'native'),
        ('4X naive (ZOH)', 4, 'zoh', 'native'), ('4X +TAC-Fold+satfix', 4, 'tac_fold_satfix', 'native'), ('4X +QP (ours)', 4, 'qp', 'native')]
for pol, name in (('dp', 'Diff.'), ('fm', 'FM')):
    print(f'\n| {name} configuration | ' + ' | '.join(hdr) + ' |'); print('|---' * (len(hdr) + 1) + '|')
    for lab, k, arm, ref in ROWS:
        cs = [(f'{rate(pol,k,t,arm):.0f}%' if rate(pol,k,t,arm) is not None else '—') if ref is None else cell(pol, k, t, arm, ref) for t in tasks]
        if any(c != '—' for c in cs): print(f'| {name} {lab} | ' + ' | '.join(cs) + ' |')
print('\ncell = success% (delta vs 1X Base, paired exact McNemar; * p<0.05). n=100 except Diff. PickCube 4X n=400.')
