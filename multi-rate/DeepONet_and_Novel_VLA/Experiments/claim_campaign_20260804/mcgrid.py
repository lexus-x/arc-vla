"""Exact McNemar on the matched-budget 30K grid, paired by (task, trial).

Only arms whose log shows a completed 300/300 are read -- a json can exist while its arm is still
writing, and q6's [-s file] check does not catch that.
"""
import json, glob, os, re, sys
from math import comb

SUITES = {"libero_spatial": "multirate_honest.json",
          "libero_object": "multirate_libero_object.json",
          "libero_10": "multirate_libero_10.json"}

def load(arm):
    d = f"grid_{arm}_out"
    if not os.path.isdir(d):
        return None
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if arm in j:
            return j[arm]
    return None

def finished(arm):
    lg = f"grid_{arm}.log"
    return os.path.exists(lg) and bool(re.search(r"^--> " + re.escape(arm) + r":", open(lg, errors="ignore").read(), re.M))

def outcomes(v):
    """-> {(task, seed): 0/1}. Seed is the pairing variable: the same seed is the same
    initial state, so a (task, seed) cell is genuinely paired across arms."""
    out = {}
    for t, blk in (v.get("per_task") or {}).items():
        for i, ep in enumerate(blk.get("episodes", [])):
            out[(str(t), ep.get("seed", i))] = int(bool(ep.get("success")))
    return out


def transforms(v):
    """The harness records how many times each rescale actually fired. A 40 Hz arm with a None
    or 0 counter silently ran unmodified -- this campaign has shipped that exact bug before."""
    d = v.get("diagnostics") or {}
    return {k: d.get(k) for k in ("spline_applied", "magscale_applied", "cadence", "anchor_applied")}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)

for suite, tag in [("libero_spatial", "spatial"), ("libero_object", "object"), ("libero_10", "long")]:
    arms = [f"v2_{tag}_native_20env", f"v2_{tag}_cadmag_spline_40env",
            f"flow30_{tag}_native_20env", f"flow30_{tag}_cadmag_spline_40env"]
    have = {a: load(a) for a in arms if finished(a)}
    if len(have) < 2:
        print(f"\n### {suite}: {len(have)}/4 arms finished -- skipping\n"); continue
    print(f"\n### {suite}  ({len(have)}/4 arms finished)")
    O = {}
    for a, v in have.items():
        o = outcomes(v)
        O[a] = o
        tr = transforms(v)
        fired = ", ".join(f"{k.replace(chr(95)+chr(97)+chr(112)+chr(112)+chr(108)+chr(105)+chr(101)+chr(100),chr(0))}" for k in [])
        note = " ".join(f"{k}={tr[k]}" for k in tr if tr[k] not in (None, 0))
        print(f"   {a:38s} n={len(o):4d}  {100*sum(o.values())/max(len(o),1):5.1f}%   {note}")
    if any(not o for o in O.values()):
        print("   !! could not extract per-trial outcomes; per-arm rates above are still valid")
        continue
    pairs = [(f"flow30_{tag}_native_20env", f"v2_{tag}_native_20env",      "flow vs v2 @20Hz"),
             (f"flow30_{tag}_cadmag_spline_40env", f"v2_{tag}_cadmag_spline_40env", "flow vs v2 @40Hz+spline"),
             (f"flow30_{tag}_native_20env", f"flow30_{tag}_cadmag_spline_40env", "flow 20Hz vs 40Hz"),
             (f"v2_{tag}_native_20env", f"v2_{tag}_cadmag_spline_40env",    "v2   20Hz vs 40Hz")]
    print(f"   {'comparison':30s} {'delta':>9} {'b/c':>10} {'p':>9}")
    for x, y, label in pairs:
        if x not in O or y not in O:
            continue
        ks = sorted(set(O[x]) & set(O[y]), key=str)
        if not ks:
            print(f"   {label:30s}   no shared (task,trial) keys"); continue
        b = sum(1 for k in ks if O[x][k] and not O[y][k])
        c = sum(1 for k in ks if O[y][k] and not O[x][k])
        d = 100 * (sum(O[x][k] for k in ks) - sum(O[y][k] for k in ks)) / len(ks)
        print(f"   {label:30s} {d:+8.1f}pp {b:4d}/{c:<4d} {mcnemar(b,c):8.4f}  (n={len(ks)})")
