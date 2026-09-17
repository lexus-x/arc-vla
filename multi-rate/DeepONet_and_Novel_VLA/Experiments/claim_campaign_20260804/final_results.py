"""Consolidated 40 Hz campaign results + paired McNemar + TOST equivalence bounds.
Runs on blackwell, next to the rollout data. Stdlib only.
"""
import json, os, glob
from math import comb

BASE = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"


def betainc(a, b, x):
    n = a + b - 1
    return sum(comb(n, k) * x**k * (1 - x) ** (n - k) for k in range(a, n + 1))


def beta_ppf(p, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp_ci(k, n, conf=0.90):
    al = 1 - conf
    lo = 0.0 if k == 0 else beta_ppf(al / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1 - al / 2, k + 1, n - k)
    return lo, hi


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def stats(b, c, n):
    d = (b - c) / n * 100
    lo_pi, hi_pi = cp_ci(b, b + c, 0.90)
    lo = (2 * lo_pi - 1) * (b + c) / n * 100
    hi = (2 * hi_pi - 1) * (b + c) / n * 100
    return d, lo, hi, mcnemar(b, c), max(abs(lo), abs(hi))


def spatial(path):
    d = json.load(open(path))
    return {k: {(t, e["seed"]): e["success"]
                for t, tv in v["per_task"].items() for e in tv["episodes"]}
            for k, v in d.items() if v.get("aggregate") is not None}


def plus(path):
    d = json.load(open(path))
    return {k: {t: e["success"] for t, e in v["per_task"].items()}
            for k, v in d.items() if v.get("aggregate") is not None}


def row(label, A, B, margin=5.0):
    ks = sorted(set(A) & set(B))
    b = sum(1 for k in ks if A[k] and not B[k])
    c = sum(1 for k in ks if B[k] and not A[k])
    d, lo, hi, p, m = stats(b, c, len(ks))
    ok = "PASS" if m <= margin else "fail"
    print(f"  {label:<38} n={len(ks):<4} {d:+5.1f}pp  {b:>3}/{c:<3} "
          f"p={p:.4f}  90%CI[{lo:+.1f},{hi:+.1f}]  +/-{m:.1f}pp {ok}")


S = spatial(f"{BASE}/spatial40_out/multirate_honest.json")
S20 = spatial(f"{BASE}/powered_out/multirate_honest.json")
P = plus(f"{BASE}/plus40_out/plus_multirate.json")
P30 = plus(f"{BASE}/plus_multirate_out/plus_multirate.json")

print("=" * 100)
print("LIBERO-SPATIAL @40 Hz  (paired n=300, init_state_id pinned)")
print("=" * 100)
for k, v in sorted(S.items(), key=lambda x: -sum(x[1].values())):
    print(f"  {k:<34} {sum(v.values())}/{len(v)} = {sum(v.values())/len(v)*100:.1f}%")
print(f"  {'asrc_native_20env (20 Hz ref)':<34} "
      f"{sum(S20['asrc_native_20env'].values())}/{len(S20['asrc_native_20env'])} = "
      f"{sum(S20['asrc_native_20env'].values())/len(S20['asrc_native_20env'])*100:.1f}%")
print("\n  PRIMARY TEST (prereg margin +/-5.0 pp):")
row("ours/folding vs spline", S["asrc_cadmag_folding_40env"], S["asrc_cadmag_spline_40env"])
row("RAI/anchor   vs spline", S["asrc_anchor_40env"], S["asrc_cadmag_spline_40env"])
print("\n  vs 20 Hz reference (secondary):")
for k in ("asrc_cadmag_folding_40env", "asrc_cadmag_spline_40env", "asrc_anchor_40env"):
    row(f"{k.replace('asrc_','')} vs native@20", S[k], S20["asrc_native_20env"])

print("\n" + "=" * 100)
print("LIBERO-PLUS @40 Hz  (paired n=105, 15/category x 7)")
print("=" * 100)
for k, v in sorted(P.items(), key=lambda x: -sum(x[1].values())):
    print(f"  {k:<34} {sum(v.values())}/{len(v)} = {sum(v.values())/len(v)*100:.1f}%")
print(f"  {'asrc_native_20env (20 Hz ref)':<34} "
      f"{sum(P30['asrc_native_20env'].values())}/105 = "
      f"{sum(P30['asrc_native_20env'].values())/105*100:.1f}%")
print()
row("ours/folding vs spline", P["asrc_cadmag_folding_40env"], P["asrc_cadmag_spline_40env"])
row("ours/folding vs native@20", P["asrc_cadmag_folding_40env"], P30["asrc_native_20env"])
row("spline       vs native@20", P["asrc_cadmag_spline_40env"], P30["asrc_native_20env"])
row("RAI/anchor   vs native@20", P["asrc_anchor_40env"], P30["asrc_native_20env"])
