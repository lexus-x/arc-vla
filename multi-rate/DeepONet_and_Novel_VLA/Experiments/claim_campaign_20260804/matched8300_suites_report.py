"""Matched-8300 ASRC vs flow report for object/goal/long suites (phase 2).
Runs on blackwell next to the campaign. Stdlib only."""
import json, glob, math

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"


def betainc(a, b, x):
    n = a + b - 1
    return sum(math.comb(n, k) * x**k * (1 - x)**(n - k) for k in range(a, n + 1))


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
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def wilson(k, n, z=1.645):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / den, (c + m) / den)


def paired_block(label, pairs):
    b = sum(1 for _, a, bb in pairs if a and not bb)
    c = sum(1 for _, a, bb in pairs if not a and bb)
    k = len(pairs)
    sa = sum(1 for _, a, _ in pairs if a)
    sb = sum(1 for _, _, bb in pairs if bb)
    print(f"  [{label}] pairs={k}  a={100*sa/k:.1f}% b={100*sb/k:.1f}%  "
          f"delta={100*(sa-sb)/k:+.1f} pp  discordant a-only={b} b-only={c}  "
          f"exact McNemar p={mcnemar(b, c):.4f}")
    if b + c > 0:
        lo, hi = cp_ci(b, b + c)
        print(f"      delta 90% CI (from discordants): "
              f"[{100*(2*lo-1)*(b+c)/k:+.1f}, {100*(2*hi-1)*(b+c)/k:+.1f}] pp")


SUITES = [("object", "asrc_object_native_20env", "flow8300_object_native_20env"),
          ("goal", "asrc_goal_native_20env", "flow8300_goal_native_20env"),
          ("long", "asrc_long_native_20env", "flow8300_long_native_20env")]

print("=" * 96)
print("LIBERO OBJECT / GOAL / LONG native 20 Hz, MATCHED 8300 steps (n=50 per arm)")
print("=" * 96)
for tag, a_arm, f_arm in SUITES:
    cands = glob.glob(f"{CAMPAIGN}/matched8300_{tag}_out/multirate_libero_*.json")
    if not cands:
        print(f"\n{tag.upper()}: no results file yet")
        continue
    d = json.load(open(cands[0]))
    print(f"\n{tag.upper()}  ({cands[0].split('/')[-1]})")
    for a in (a_arm, f_arm):
        e = d.get(a, {})
        if e.get("aggregate") is None:
            print(f"  {a:30s} MISSING")
            continue
        n = e.get("diagnostics", {}).get("n")
        lo, hi = wilson(int(round(e["aggregate"] * n)), n) if n else (float("nan"), float("nan"))
        print(f"  {a:30s} {100*e['aggregate']:5.1f}%  (n={n})"
              + (f"  [90% CI {100*lo:.1f}, {100*hi:.1f}]" if n else ""))
    if all(d.get(a, {}).get("aggregate") is not None for a in (a_arm, f_arm)):
        p0, p1 = d[a_arm]["per_task"], d[f_arm]["per_task"]
        pairs = []
        for tk in sorted(set(p0) & set(p1)):
            ep0 = {ep["seed"]: ep["success"] for ep in p0[tk]["episodes"]}
            ep1 = {ep["seed"]: ep["success"] for ep in p1[tk]["episodes"]}
            for seed in sorted(set(ep0) & set(ep1)):
                pairs.append((f"task{tk}/{seed}", ep0[seed], ep1[seed]))
        paired_block(f"{tag}: asrc vs flow8300", pairs)
print("\nDONE")
