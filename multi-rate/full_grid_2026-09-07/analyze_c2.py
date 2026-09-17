"""Campaign 2 (PLAN_BLOCKSUM_HEAD): seed-averaged table + pre-registered contrasts from result_dp_*_c2.json."""
import json, glob, re, collections, numpy as np
from scipy.stats import binomtest
R = collections.defaultdict(dict)  # (task, head, k, arm) -> seed -> success list
for f in glob.glob('result_dp_*_c2.json'):
    m = re.match(r'result_dp_(.+?)(?:_(bspline|blocksum(?:_b\d+)?))?(?:_s(\d))?(?:_k(\d))?_c2\.json', f)
    task, head, seed, k = m.group(1), m.group(2) or 'step', int(m.group(3) or 0), int(m.group(4) or 2)
    for arm, v in json.load(open(f))['success'].items(): R[(task, head, k, arm)][seed] = v
def mcn(a, b):
    x = sum(1 for p, q in zip(a, b) if p and not q); y = sum(1 for p, q in zip(a, b) if q and not p)
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)
def holm(ps):
    o = np.argsort(ps); adj = np.zeros(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
ROWS = [('step', 'native', 'Base (per-step)'), ('step', 'zoh', 'Base, naive speedup (ZOH)'), ('step', 'qp', 'Base + post-hoc QP'),
        ('bspline', 'native', 'B-spline head'), ('blocksum', 'zoh', 'Block-sum head, ZOH decode'), ('blocksum', 'qp', 'Block-sum head + QP (ours)')]
for task in sorted({t for t, *_ in R}):
    print(f'\n### {task}  (success %, mean ± sd over seeds; n per seed in brackets)')
    print('| Configuration | 1X | 2X | 4X |'); print('|---|---|---|---|')
    for head, arm, lab in ROWS:
        cs = []
        for k in (1, 2, 4):
            d = R.get((task, head, k, arm), {})
            if not d: cs.append('—'); continue
            m = [100 * np.mean(v) for v in d.values()]
            cs.append(f'{np.mean(m):.0f} ± {np.std(m):.0f} [{len(m)}s×{len(next(iter(d.values())))}]')
        print(f'| {lab} | ' + ' | '.join(cs) + ' |')
    def contrast(name, k, a, b):
        da, db = R.get((task,) + a, {}), R.get((task,) + b, {}); seeds = sorted(set(da) & set(db))
        if not seeds: print(f'  {name}: no data'); return
        ps, lines = [], []
        for s in seeds:
            x, y, p = mcn(da[s], db[s]); ps.append(p)
            lines.append(f's{s}: {100*(np.mean(da[s])-np.mean(db[s])):+.1f}pp {x}/{y} p={p:.3g}')
        print(f'  {name} @{k}X: ' + ' | '.join(lines) + f' | Holm min={holm(ps).min():.3g}')
    print('pre-registered contrasts:')
    contrast('P1 ours vs B-spline head', 4, ('blocksum', 4, 'qp'), ('bspline', 4, 'native'))
    contrast('P2 ours vs B-spline head', 2, ('blocksum', 2, 'qp'), ('bspline', 2, 'native'))
    contrast('P3 ours vs Base', 1, ('blocksum', 1, 'qp'), ('step', 1, 'native'))
    contrast('P3 ours vs B-spline', 1, ('blocksum', 1, 'qp'), ('bspline', 1, 'native'))
    contrast('P4a QP vs ZOH decode', 4, ('blocksum', 4, 'qp'), ('blocksum', 4, 'zoh'))
    contrast('P4b ours vs Base+post-hoc QP', 4, ('blocksum', 4, 'qp'), ('step', 4, 'qp'))
    contrast('ref ours vs naive speedup', 4, ('blocksum', 4, 'qp'), ('step', 4, 'zoh'))
