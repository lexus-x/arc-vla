"""Matched-8300 ASRC vs flow report: LIBERO-Plus + libero_spatial, paired stats.
Runs on blackwell next to the campaign. Stdlib only."""
import json, glob, math, sys

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
A_ASRC = "asrc_native_20env"
A_FLOW = "flow8300_native_20env"


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


def paired_block(label, pairs, n_a, n_b):
    """pairs: list of (task_key, a_success, b_success); n_a/n_b: episode counts."""
    b = sum(1 for _, a, bb in pairs if a and not bb)
    c = sum(1 for _, a, bb in pairs if not a and bb)
    k = len(pairs)
    sa = sum(1 for _, a, _ in pairs if a)
    sb = sum(1 for _, _, bb in pairs if bb)
    d = sa - sb
    print(f"  [{label}] paired pairs={k}  a={100*sa/k:.1f}% b={100*sb/k:.1f}%  "
          f"delta={100*d/k:+.1f} pp  discordant a-only={b} b-only={c}")
    print(f"      exact McNemar p = {mcnemar(b, c):.4f}")
    if b + c > 0:
        pi_lo, pi_hi = cp_ci(b, b + c)
        dlo = (2 * pi_lo - 1) * (b + c)
        dhi = (2 * pi_hi - 1) * (b + c)
        print(f"      delta 90% CI (from discordants, TOST-style): "
              f"[{100*dlo/k:+.1f}, {100*dhi/k:+.1f}] pp")
    return b, c, k


def main():
    # ---------- LIBERO-Plus ----------
    p = f"{CAMPAIGN}/plus_matched8300_out/plus_multirate.json"
    d = json.load(open(p))
    print("=" * 96)
    print("LIBERO-PLUS native 20 Hz, MATCHED 8300 steps (n=105, 15/category x 7)")
    print("=" * 96)
    for a in (A_ASRC, A_FLOW):
        e = d.get(a, {})
        if e.get("aggregate") is None:
            print(f"  {a:26s} MISSING (incomplete)")
            continue
        pt = e["per_task"]
        ok = sum(v["success"] for v in pt.values())
        lo, hi = wilson(ok, len(pt))
        print(f"  {a:26s} {ok}/{len(pt)} = {100*e['aggregate']:5.1f}%  "
              f"[90% CI {100*lo:.1f}, {100*hi:.1f}]")
        for cat, r in (e.get("by_category") or {}).items():
            if r is not None:
                print(f"      {cat:24s} {100*r:5.1f}%")
    if all(d.get(a, {}).get("aggregate") is not None for a in (A_ASRC, A_FLOW)):
        p0, p1 = d[A_ASRC]["per_task"], d[A_FLOW]["per_task"]
        keys = sorted(set(p0) & set(p1))
        pairs = [(k, p0[k]["success"], p1[k]["success"]) for k in keys]
        paired_block(f"Plus: {A_ASRC} vs {A_FLOW}", pairs, len(p0), len(p1))

    # ---------- libero_spatial ----------
    print()
    print("=" * 96)
    print("LIBERO-SPATIAL native 20 Hz, MATCHED 8300 steps (30 trials x 10 tasks = n=300)")
    print("=" * 96)
    cands = glob.glob(f"{CAMPAIGN}/matched8300_spatial_out/multirate_libero_spatial.json")
    if not cands:
        print("  no results file yet")
    else:
        d2 = json.load(open(cands[0]))
        for a in (A_ASRC, A_FLOW):
            e = d2.get(a, {})
            if e.get("aggregate") is None:
                print(f"  {a:26s} MISSING (incomplete)")
                continue
            print(f"  {a:26s} {100*e['aggregate']:5.1f}%  (n={e.get('diagnostics',{}).get('n','?')})")
            for tk, tv in sorted(e["per_task"].items()):
                print(f"      task {tk:>2s} {100*tv['success_rate']:5.1f}%")
        if all(d2.get(a, {}).get("aggregate") is not None for a in (A_ASRC, A_FLOW)):
            p0, p1 = d2[A_ASRC]["per_task"], d2[A_FLOW]["per_task"]
            pairs = []
            for tk in sorted(set(p0) & set(p1)):
                ep0 = {ep["seed"]: ep["success"] for ep in p0[tk]["episodes"]}
                ep1 = {ep["seed"]: ep["success"] for ep in p1[tk]["episodes"]}
                for seed in sorted(set(ep0) & set(ep1)):
                    pairs.append((f"task{tk}/seed{seed}", ep0[seed], ep1[seed]))
            paired_block(f"Spatial: {A_ASRC} vs {A_FLOW}", pairs, None, None)
    print()
    print("DONE")


if __name__ == "__main__":
    main()
