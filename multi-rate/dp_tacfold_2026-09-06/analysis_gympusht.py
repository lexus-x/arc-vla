"""gym-pusht results -> paper row. Run after harness finishes."""
import json
import math

D = "/home/user/Desktop/multi-rate/dp_tacfold_2026-09-06"
r = json.load(open(f"{D}/result_gym-pusht.json"))

S = {a: r["success"][a] for a in r["arms"]}
n = r["n"]
print(f"gym-pusht closed-loop DP | k={r['k']} | n={n} | train_steps={r['train_steps']}")
print(f"policy raw sat frac: {r['policy_raw_sat_frac']['native']:.3f}")
print()
print(f"{'arm':<18} {'succ':>10}", end="")
for ref in ("zoh", "native"):
    print(f"  vs {ref}: d_pp / p", end="")
print()
rows = []
for a in r["arms"]:
    line = f"{a:<18} {100*sum(S[a])/n:>6.1f}%   "
    for ref in ("zoh", "native"):
        if a == ref:
            line += "  -"
            continue
        c = r["contrasts"][a][f"vs_{ref}"]
        line += f"  {c['delta_pp']:+5.1f}pp p={c['p']:.3g}"
        rows.append((a, ref, c["delta_pp"], c["p"]))
    print(line)

# Holm across all non-trivial contrasts
ps = sorted((p, a, ref, d) for a, ref, d, p in rows if p < 1.0)
m = len(ps)
print(f"\nHolm over {m} contrasts:")
for rank, (p, a, ref, d) in enumerate(ps):
    holm = min(1.0, p * (m - rank))
    flag = "SIG" if holm < 0.05 else ""
    print(f"  {a} vs {ref}: raw={p:.4g} holm={holm:.4g} {flag}")
