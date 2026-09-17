"""Paired exact McNemar, replan=5 / 520 steps, v2 vs flow (state-history FIXED)."""
import json, glob, os
from math import comb
def load(d, arm):
    for f in glob.glob(os.path.join(d, "*.json")):
        j = json.load(open(f))
        if arm in j: return j[arm]
    return None
def outc(v):
    o = {}
    for t, b in (v.get("per_task") or {}).items():
        for i, e in enumerate(b.get("episodes", [])):
            o[(str(t), e.get("seed", i))] = int(bool(e.get("success")))
    return o
def mc(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k+1)) / 2**n)
print(f"{'suite':9s} {'v2':>7} {'flow':>7} {'v2-flow':>9} {'v2w/flw':>10} {'p':>9}")
print("-" * 58)
for suite, tag in (("libero_spatial","spatial"),("libero_object","object"),("libero_10","long")):
    a = load(f"r5g_v2_{tag}_native_20env_out", f"v2_{tag}_native_20env")
    b_ = load(f"r5g_flow30_{tag}_native_20env_out", f"flow30_{tag}_native_20env")
    if not a or not b_:
        print(f"{tag:9s}  pending"); continue
    A, B = outc(a), outc(b_)
    ks = sorted(set(A) & set(B), key=str)
    w = sum(1 for k in ks if A[k] and not B[k])   # v2 wins
    l = sum(1 for k in ks if B[k] and not A[k])   # flow wins
    va = 100*sum(A[k] for k in ks)/len(ks); fa = 100*sum(B[k] for k in ks)/len(ks)
    print(f"{tag:9s} {va:6.1f}% {fa:6.1f}% {va-fa:+8.1f}pp {w:4d}/{l:<5d} {mc(w,l):8.4f} (n={len(ks)})")
