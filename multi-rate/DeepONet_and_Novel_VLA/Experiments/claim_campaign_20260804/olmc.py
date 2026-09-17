"""Paired exact McNemar, open loop (replan=50 = full chunk), v2 (state-history FIXED) vs flow."""
import json, glob, os
from math import comb

def load(d, arm):
    for f in glob.glob(os.path.join(d, "*.json")):
        j = json.load(open(f))
        if arm in j:
            return j[arm]
    return None

def outcomes(v):
    out = {}
    for t, blk in (v.get("per_task") or {}).items():
        for i, ep in enumerate(blk.get("episodes", [])):
            out[(str(t), ep.get("seed", i))] = int(bool(ep.get("success")))
    return out

def mc(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)

print(f"{'suite':9s} {'v2':>7} {'flow':>7} {'delta':>9} {'b/c':>10} {'p':>9}")
print("-" * 56)
for suite, tag in (("libero_spatial", "spatial"), ("libero_object", "object"), ("libero_10", "long")):
    v2 = load(f"ol2_v2_{tag}_native_20env_out", f"v2_{tag}_native_20env")
    fl = load(f"ol_flow30_{tag}_native_20env_out", f"flow30_{tag}_native_20env")
    if not v2 or not fl:
        print(f"{tag:9s}  missing"); continue
    A, B = outcomes(v2), outcomes(fl)
    ks = sorted(set(A) & set(B), key=str)
    b = sum(1 for k in ks if B[k] and not A[k])   # flow wins
    c = sum(1 for k in ks if A[k] and not B[k])   # v2 wins
    va, fa = 100*sum(A[k] for k in ks)/len(ks), 100*sum(B[k] for k in ks)/len(ks)
    print(f"{tag:9s} {va:6.1f}% {fa:6.1f}% {fa-va:+8.1f}pp {b:4d}/{c:<4d} {mc(b,c):8.4f}  (n={len(ks)})")
