import json, numpy as np
R = {k: json.load(open(f"qp_replay_PickCube-v1_k{k}.json")) for k in (2,3,4,5)}
KS = json.load(open("ksweep_results.json"))
ARMS = ["original","exact_integral","cubic_spline_satfix","pchip_satfix","tac_fold_satfix","qp"]
def holm(items):  # items: list of (label, p)
    order = sorted(range(len(items)), key=lambda i: items[i][1]); m=len(items); out=[None]*m; run=0.0
    for rank,i in enumerate(order):
        adj=min(1.0,(m-rank)*items[i][1]); run=max(run,adj); out[i]=(items[i][0],items[i][1],run,run<0.05)
    return out
print("=== SUCCESS RATE, PickCube-v1 open-loop paired replay, n=993 per k ===")
print(f"{'arm':22s}" + "".join(f"{'k='+str(k):>10s}" for k in (2,3,4,5)))
for a in ARMS:
    print(f"{a:22s}" + "".join(f"{100*np.mean(R[k]['success'][a]):9.2f}%" for k in (2,3,4,5)))
print("\n=== qp contrasts (delta pp, wins/losses, raw p, 95% CI) ===")
for k in (2,3,4,5):
    for ref in ("exact_integral","cubic_spline_satfix","tac_fold_satfix","pchip_satfix"):
        c = R[k]["contrasts"][f"qp_vs_{ref}"]
        print(f"k={k} qp vs {ref:20s} {c['delta_pp']:+6.2f}pp  {c['wins']:3d}/{c['losses']:3d}  p={c['p']:.3g}  CI=[{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]")
# P1 primary family m=3
prim = [(f"k={k}", R[k]["contrasts"]["qp_vs_cubic_spline_satfix"]["p"]) for k in (3,4,5)]
ph = holm(prim)
d4 = R[4]["contrasts"]["qp_vs_cubic_spline_satfix"]["delta_pp"]
p1_holm = [x for x in ph if x[0]=="k=4"][0]
print("\n=== P1 PRIMARY: qp vs cubic_spline_satfix @k=4 (need >= +3pp AND Holm m=3) ===")
for lbl,p,ph_,s in ph: print(f"  {lbl}: raw p={p:.3g} Holm p={ph_:.3g} survives={s}")
print(f"  delta@k=4 = {d4:+.2f}pp  ->  P1 {'HOLDS' if (d4>=3 and p1_holm[3]) else 'FAILS'}")
# secondary m=16
sec=[]
for k in (2,3,4,5):
    for ref in ("exact_integral","cubic_spline_satfix","pchip_satfix","tac_fold_satfix"):
        sec.append((f"k={k} qp_vs_{ref}", R[k]["contrasts"][f"qp_vs_{ref}"]["p"]))
sh = holm(sec)
print("\n=== SECONDARY FAMILY Holm m=16 ===")
for lbl,p,ph_,s in sorted(sh,key=lambda x:x[1]):
    k=int(lbl[2]); ref=lbl.split("qp_vs_")[1]; d=R[k]["contrasts"][f"qp_vs_{ref}"]["delta_pp"]
    print(f"  {lbl:32s} {d:+6.2f}pp raw p={p:.3g} Holm p={ph_:.3g} {'SURVIVES' if s else ''}")
# P2-P5
print("\n=== P2: qp beats ZOH at every k (Holm) ===")
for k in (2,3,4,5):
    x=[y for y in sh if y[0]==f"k={k} qp_vs_exact_integral"][0]; print(f"  k={k}: {R[k]['contrasts']['qp_vs_exact_integral']['delta_pp']:+.2f}pp Holm p={x[2]:.3g} -> {'ok' if x[3] and R[k]['contrasts']['qp_vs_exact_integral']['delta_pp']>0 else 'FAIL'}")
c2=R[2]["contrasts"]; print(f"\n=== P3 @k=2: qp vs cubic {c2['qp_vs_cubic_spline_satfix']['delta_pp']:+.2f}pp p={c2['qp_vs_cubic_spline_satfix']['p']:.3g} (need |d|<=2, n.s.); qp vs tac_fold {c2['qp_vs_tac_fold_satfix']['delta_pp']:+.2f}pp (need <0) ===")
print("=== P4: qp >= tac_fold_satfix at k>=3, margin growing ===")
for k in (3,4,5): print(f"  k={k}: {R[k]['contrasts']['qp_vs_tac_fold_satfix']['delta_pp']:+.2f}pp p={R[k]['contrasts']['qp_vs_tac_fold_satfix']['p']:.3g}")
print("=== P5: MSE ranking (ksweep) vs success ranking, per k ===")
mmap={"exact_integral":"zoh","cubic_spline_satfix":"spline_satfix","pchip_satfix":"pchip_satfix","tac_fold_satfix":"tac_fold_satfix","qp":"qp"}
for k in (2,3,4,5):
    ms=KS[f"PickCube_k{k}"]; mse_rank=sorted(mmap, key=lambda a: ms[mmap[a]]); sr_rank=sorted(mmap, key=lambda a: -np.mean(R[k]["success"][a]))
    print(f"  k={k} MSE best->worst: {[mmap[a] for a in mse_rank]}\n       SR  best->worst: {[mmap[a] for a in sr_rank]}  match={mse_rank==sr_rank}")
