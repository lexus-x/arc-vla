"""Consolidated multi-rate report across every benchmark run, with diagnostics.

Prints, per benchmark: each arm's rate, and for the ours-vs-spline pair a paired exact
McNemar on the identical task/trial list. Then a diagnostics block showing WHERE failures
happen (timeout at the wall-clock cap vs early termination), whether each intervention
actually fired, and the observation-cadence counters. That is the part that tells you if a
zero is a real loss or a broken arm.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

OUT = Path("/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
BASE = OUT / "multirate_honest_out"
PLUS = OUT / "plus_multirate_out" / "plus_multirate.json"
SUITES = [("libero_spatial", BASE / "multirate_honest.json"),
          ("libero_object", BASE / "multirate_libero_object.json"),
          ("libero_goal", BASE / "multirate_libero_goal.json"),
          ("libero_10", BASE / "multirate_libero_10.json")]
SPLINE = "asrc_cadmag_spline_30env"
# Every candidate for "ours" is paired against the SAME spline arm on the overlapping
# (task, seed) pairs, so a new arm never gets read against a different trial set.
OURS_ARMS = ["asrc_cadmag_folding_30env", "asrc_anchor_30env"]
OURS = OURS_ARMS[0]


def mcnemar(pairs):
    """pairs: list of (ours_success, spline_success). Returns (b, c, p)."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    d = b + c
    if not d:
        return b, c, 1.0
    p = sum(comb(d, k) for k in range(0, min(b, c) + 1)) / 2 ** d * 2
    return b, c, min(1.0, p)


def episodes(entry):
    return [e for t in entry.get("per_task", {}).values() for e in t.get("episodes", [])]


def report_libero(name, path):
    if not path.exists():
        print(f"\n### {name}: not run")
        return
    d = json.loads(path.read_text())
    print(f"\n### {name}   ({path.name})")
    rows = []
    for k in sorted(d, key=lambda k: -(d[k].get("aggregate") or -1)):
        e = d[k]
        eps = episodes(e)
        if not eps:
            continue
        ok = sum(x["success"] for x in eps)
        cfg = e.get("config", {})
        rows.append((k, ok, len(eps), cfg.get("env_freq"), cfg.get("transform")))
    for k, ok, n, hz, tr in rows:
        print(f"  {k:30s} {ok:3d}/{n:3d} = {100*ok/n:5.1f}%   {str(hz):>2s} Hz  {tr or '-'}")

    for ours in OURS_ARMS:
        if ours not in d or SPLINE not in d:
            continue
        pa, pb = d[ours]["per_task"], d[SPLINE]["per_task"]
        pairs = []
        for t in sorted(set(pa) & set(pb), key=int):
            ea, eb = pa[t]["episodes"], pb[t]["episodes"]
            for i in range(min(len(ea), len(eb))):
                if ea[i]["seed"] == eb[i]["seed"]:
                    pairs.append((ea[i]["success"], eb[i]["success"]))
        if pairs:
            b, c, p = mcnemar(pairs)
            delta = 100 * (b - c) / len(pairs)
            print(f"  PAIRED {ours} vs spline on {len(pairs)} matched rollouts: "
                  f"{delta:+.1f} pp | ours-only {b} spline-only {c} | McNemar p={p:.4f}")

    print("  diagnostics (where failures land):")
    for k in sorted(d):
        g = d[k].get("diagnostics")
        if not g:
            continue
        cad = g.get("cadence") or {}
        print(f"    {k:30s} cap={g['cap']:3d} at_cap={g['fail_at_cap']:3d} early={g['fail_early']:3d} "
              f"succ_med={g['success_steps_median']} | spline={g['spline_applied']} "
              f"mag={g['magscale_applied']} q={cad.get('queries')} res={cad.get('resolved')} "
              f"starved={cad.get('starved')}")


def report_plus():
    if not PLUS.exists():
        print("\n### LIBERO-Plus: not run")
        return
    d = json.loads(PLUS.read_text())
    print(f"\n### LIBERO-Plus   ({PLUS.name})")
    for k in sorted(d, key=lambda k: -(d[k].get("aggregate") or -1)):
        pt = d[k].get("per_task", {})
        if not pt:
            continue
        ok = sum(v["success"] for v in pt.values())
        n = len(pt)
        cfg = d[k].get("config", {})
        flag = "" if n >= 105 else f"  [PARTIAL {n}/105]"
        print(f"  {k:30s} {ok:3d}/{n:3d} = {100*ok/n:5.1f}%   {cfg.get('env_freq')} Hz{flag}")
    if OURS in d and SPLINE in d:
        pa, pb = d[OURS]["per_task"], d[SPLINE]["per_task"]
        keys = sorted(set(pa) & set(pb), key=int)
        pairs = [(pa[k]["success"], pb[k]["success"]) for k in keys]
        b, c, p = mcnemar(pairs)
        print(f"  PAIRED ours vs spline on {len(keys)} matched tasks: "
              f"{100*(b-c)/len(keys):+.1f} pp | ours-only {b} spline-only {c} | McNemar p={p:.4f}")
        cats = {}
        for k in keys:
            s = cats.setdefault(pa[k]["category"], [0, 0, 0])
            s[0] += pa[k]["success"]; s[1] += pb[k]["success"]; s[2] += 1
        print("  per category (ours vs spline of n):")
        for cat in sorted(cats):
            o, s, n = cats[cat]
            tag = "ours" if o > s else ("spline" if s > o else "tie")
            zero = "  <- BOTH ZERO: verify the category is hard, not broken" if o == s == 0 else ""
            print(f"    {cat:22s} {o:2d} vs {s:2d} of {n:2d}  {tag}{zero}")


if __name__ == "__main__":
    print("=" * 78)
    print("MULTI-RATE CONSOLIDATED REPORT")
    print("  protocol: genuine sim control_freq | matched wall-clock 11.0 s |")
    print("            20 Hz decision cadence (phase accumulator) | raw-space magnitude")
    print("  checkpoints trained on libero_spatial ONLY -> object/goal/10 are cross-suite OOD")
    print("  MetaWorld omitted: action (4,) vs policy 7, obs 39-dim stateless, untrained robot")
    print("=" * 78)
    for name, path in SUITES:
        report_libero(name, path)
    report_plus()
    print("\n" + "=" * 78)
