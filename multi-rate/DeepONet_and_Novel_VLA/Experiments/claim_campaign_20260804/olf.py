"""Paired exact McNemar: folding vs spline, 40 Hz OPEN LOOP (replan_steps=100), same asrc ckpt."""
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
F = load("olf_asrc_cadmag_folding_40env_out", "asrc_cadmag_folding_40env")
S = load("olf_asrc_cadmag_spline_40env_out", "asrc_cadmag_spline_40env")
A, B = outc(F), outc(S)
ks = sorted(set(A) & set(B), key=str)
fw = sum(1 for k in ks if A[k] and not B[k])
sw = sum(1 for k in ks if B[k] and not A[k])
fa = 100*sum(A[k] for k in ks)/len(ks); sa = 100*sum(B[k] for k in ks)/len(ks)
print(f"n paired            : {len(ks)}")
print(f"folding             : {fa:.1f}%")
print(f"spline              : {sa:.1f}%")
print(f"delta (fold-spline) : {fa-sa:+.1f} pp")
print(f"discordant fold/spl : {fw}/{sw}")
print(f"exact McNemar p     : {mc(fw,sw):.4f}")
print()
print("PRE-REGISTERED GATE: positive delta AND p < 0.05")
print("VERDICT:", "PASS" if (fa > sa and mc(fw,sw) < 0.05) else "FAIL -- kill rule fires")
# transform counters: prove neither arm silently no-oped
for tag, v in (("folding", F), ("spline", S)):
    d = v.get("diagnostics") or {}
    print(f"  {tag:8s} spline_applied={d.get('spline_applied')} magscale_applied={d.get('magscale_applied')} cap={d.get('cap')}")
