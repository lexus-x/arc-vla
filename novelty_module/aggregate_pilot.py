"""aggregate_pilot.py — build the paired pilot table from results/pilot_results.json.

Per category and arm: success rate; vs base: raw delta, paired discordant counts
(ours-only vs base-only wins) and exact McNemar p-value (binomial, two-sided).
"""
import json, math, sys
from pathlib import Path

RES = Path("/home/user/Desktop/novelty_module/results/pilot_results.json")
ARMS = ["canon", "consensus", "canon+consensus"]


def mcnemar(b_only, a_only):
    n = b_only + a_only
    if n == 0:
        return 1.0
    k = min(b_only, a_only)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n * 2
    return min(1.0, p)


r = json.loads(RES.read_text())
cats = r["_config"]["categories"]
base = r["arms"]["base"]
print(f"{'category':22s} " + " ".join(f"{a:>28s}" for a in ARMS))
for c in cats:
    cells = []
    for arm in ARMS:
        arm_d = r["arms"].get(arm, {}).get(c, {}).get("per_task", {})
        base_d = base.get(c, {}).get("per_task", {})
        if not arm_d:
            cells.append(f"{'--':>28s}")
            continue
        keys = sorted(set(arm_d) & set(base_d))
        a_s = sum(arm_d[k]["success"] for k in keys)
        b_s = sum(base_d[k]["success"] for k in keys)
        a_only = sum(1 for k in keys if arm_d[k]["success"] and not base_d[k]["success"])
        b_only = sum(1 for k in keys if not arm_d[k]["success"] and base_d[k]["success"])
        rate = sum(v["success"] for v in arm_d.values()) / len(arm_d)
        p = mcnemar(b_only, a_only)
        cells.append(f"{rate:.2f} (d={rate - b_s/len(keys):+.2f}, {a_only}/{b_only}, p={p:.2f})")
    print(f"{c:22s} " + " ".join(f"{x:>28s}" for x in cells))
