import json
from math import comb

def load(p):
    d = json.load(open(p))
    k = next(iter(d))
    return d[k]

A = load("/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/eart150_rp5_out/multirate_honest.json")
B = load("/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/spline150_rp5_out/multirate_honest.json")

b = c = d = 0
for t in range(10):
    ea = A["per_task"][str(t)]["episodes"]
    eb = B["per_task"][str(t)]["episodes"]
    for x, y in zip(ea, eb):
        xs, ys = bool(x["success"]), bool(y["success"])
        if xs and not ys: b += 1
        elif ys and not xs: c += 1
        else: d += 1
n = b + c
k = min(b, c)
p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n if n else 1.0
sa = sum(1 for t in range(10) for e in A["per_task"][str(t)]["episodes"] if e["success"])
sb = sum(1 for t in range(10) for e in B["per_task"][str(t)]["episodes"] if e["success"])
tot = sum(len(A["per_task"][str(t)]["episodes"]) for t in range(10))
print(f"n={tot}: EART {sa}/{tot}={sa/tot*100:.1f}% vs spline {sb}/{tot}={sb/tot*100:.1f}%  (+{(sa-sb)/tot*100:.1f}pp)")
print(f"paired: EART-only={b} spline-only={c} agree={d}")
print(f"McNemar n={n} k={k} p={p:.4f}")
# per-task EART
for t in range(10):
    e = A["per_task"][str(t)]["episodes"]
    s = sum(1 for x in e if x["success"])
    print(f"  t{t}: {s}/{len(e)} = {s/len(e)*100:.0f}%")
