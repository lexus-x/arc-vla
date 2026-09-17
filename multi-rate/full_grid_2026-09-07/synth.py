import json, glob

FILES = sorted(glob.glob("result_*.json"))
data = {f: json.load(open(f)) for f in FILES}

def holm(pvals, alpha=0.05):
    """pvals: list of (label, p). Returns list of (label, p, p_holm, survives)."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i][1])
    m = len(pvals)
    out = [None] * m
    running_max = 0.0
    for rank, i in enumerate(idx):
        label, p = pvals[i]
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        out[i] = (label, p, running_max, running_max < alpha)
    return out

print("=" * 100)
print("FULL PER-ARM TABLE (all 14 results)")
print("=" * 100)
for f in FILES:
    r = data[f]
    print(f"\n--- {f}: task={r['task']} policy={r.get('policy','dp')} k={r['k']} n={r['n']} train_steps={r['train_steps']} ---")
    print(f"  policy raw |a|>1 frac (native): {r['policy_raw_sat_frac']['native']:.3f}")
    for a in r["arms"]:
        s = r["success"][a]
        succ_pct = 100 * sum(s) / len(s)
        c = r["contrasts"].get(a, {})
        vz = c.get("vs_zoh", {})
        vn = c.get("vs_native", {})
        vz_str = f"vs_zoh Δ={vz.get('delta_pp',0):+.1f}pp p={vz.get('p',1):.4g}" if vz else ""
        vn_str = f"vs_native Δ={vn.get('delta_pp',0):+.1f}pp p={vn.get('p',1):.4g}" if vn else ""
        print(f"  {a:16s} {succ_pct:5.1f}% ({sum(s)}/{len(s)})  {vz_str:38s} {vn_str}")

# ---- Family 1: bspline-addition to the 5 already-complete DP/RoboMimic+ManiSkill tasks
FAM1_TASKS = ["dp_lift", "dp_can", "dp_square", "dp_PushT-v1", "dp_PickCube-v1"]
fam1 = []
for tag in FAM1_TASKS:
    r = data[f"result_{tag}.json"]
    for arm in ["bspline", "bspline_satfix"]:
        for ref in ("vs_zoh", "vs_native"):
            c = r["contrasts"][arm][ref]
            fam1.append((f"{tag}:{arm}:{ref}", c["p"], c["delta_pp"]))

print("\n" + "=" * 100)
print(f"FAMILY 1 -- bspline-addition to already-complete DP tasks, Holm over {len(fam1)} contrasts")
print("=" * 100)
res1 = holm([(lbl, p) for lbl, p, d in fam1])
d_by_lbl = {lbl: d for lbl, p, d in fam1}
for lbl, p, p_holm, surv in sorted(res1, key=lambda x: x[1]):
    flag = "  <-- SURVIVES HOLM" if surv else ""
    print(f"  {lbl:35s} delta={d_by_lbl[lbl]:+6.1f}pp  p={p:.4g}  p_holm={p_holm:.4g}{flag}")

# ---- Family 2: all new cells (FM on the 5 ManiSkill/RoboMimic tasks + all 4 RoboCasa results)
FAM2_TASKS = ["fm_lift", "fm_can", "fm_square", "fm_PushT-v1", "fm_PickCube-v1",
              "dp_RC-OpenDrawer", "fm_RC-OpenDrawer", "dp_RC-PnPCounterToStove", "fm_RC-PnPCounterToStove"]
fam2 = []
for tag in FAM2_TASKS:
    r = data[f"result_{tag}.json"]
    for arm in r["arms"]:
        if arm == "native":
            continue
        for ref in ("vs_zoh", "vs_native"):
            if arm == ref.replace("vs_", ""):
                continue
            c = r["contrasts"].get(arm, {}).get(ref)
            if c is None:
                continue
            fam2.append((f"{tag}:{arm}:{ref}", c["p"], c["delta_pp"]))

print("\n" + "=" * 100)
print(f"FAMILY 2 -- all new cells (FM x 5 tasks + RoboCasa dp+fm x 2 tasks), Holm over {len(fam2)} contrasts")
print("=" * 100)
res2 = holm([(lbl, p) for lbl, p, d in fam2])
d_by_lbl2 = {lbl: d for lbl, p, d in fam2}
survivors = [x for x in res2 if x[3]]
print(f"\n{len(survivors)} of {len(fam2)} contrasts survive Holm at alpha=0.05:")
for lbl, p, p_holm, surv in sorted(survivors, key=lambda x: x[1]):
    print(f"  {lbl:40s} delta={d_by_lbl2[lbl]:+6.1f}pp  p={p:.4g}  p_holm={p_holm:.4g}  <-- SURVIVES")

print("\nAll family-2 contrasts sorted by raw p (top 25):")
for lbl, p, p_holm, surv in sorted(res2, key=lambda x: x[1])[:25]:
    flag = " <-- SURVIVES" if surv else ""
    print(f"  {lbl:40s} delta={d_by_lbl2[lbl]:+6.1f}pp  p={p:.4g}  p_holm={p_holm:.4g}{flag}")
