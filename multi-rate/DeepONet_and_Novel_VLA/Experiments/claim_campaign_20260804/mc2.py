"""Exact McNemar on LIBERO-Plus n=315, paired by task id. Runs on blackwell."""
import json, glob
from math import comb

A = {}
for p in glob.glob('pow_plus_*_out/plus_multirate.json'):
    for k, v in json.load(open(p)).items():
        A[k] = v

def outcomes(v):
    pt = v['per_task']
    if isinstance(pt, dict):
        return {str(k): int(bool(x['success'] if isinstance(x, dict) else x)) for k, x in pt.items()}
    return {str(r.get('task_id', r.get('task', i))): int(bool(r['success'])) for i, r in enumerate(pt)}

O = {k: outcomes(v) for k, v in A.items()}
for k, o in O.items():
    print(f"{k:34s} n={len(o):4d}  {100*sum(o.values())/len(o):5.1f}%")

def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)

PAIRS = [('asrc_cadmag_folding_40env', 'asrc_cadmag_spline_40env'),
         ('asrc_cadmag_folding_40env', 'flow_m1_cadmag_spline_40env'),
         ('flow_m1_cadmag_spline_40env', 'flow_m1_native_20env')]

print(f"\n{'comparison':58s} {'delta':>8} {'b/c':>10} {'p':>9}")
print('-' * 90)
for x, y in PAIRS:
    if x not in O or y not in O:
        print(f"MISSING {x} or {y}"); continue
    keys = sorted(set(O[x]) & set(O[y]))
    b = sum(1 for t in keys if O[x][t] and not O[y][t])
    c = sum(1 for t in keys if O[y][t] and not O[x][t])
    d = 100 * (sum(O[x][t] for t in keys) - sum(O[y][t] for t in keys)) / len(keys)
    print(f"{x[:26]:26s} vs {y[:28]:28s} {d:+7.1f}pp {b:4d}/{c:<4d} {exact_mcnemar(b,c):8.4f}   (paired n={len(keys)})")
