import json, math, numpy as np
def mcn(a,b):
    a,b=np.asarray(a,bool),np.asarray(b,bool); ao=int((a&~b).sum()); bo=int((~a&b).sum()); d=ao+bo
    return ao,bo,(1.0 if d==0 else float(min(1.0,2*sum(math.comb(d,i) for i in range(min(ao,bo)+1))/2**d)))
def holm(items):
    o=sorted(range(len(items)),key=lambda i:items[i][1]); m=len(items); out=[None]*m; run=0.0
    for r,i in enumerate(o):
        adj=min(1.0,(m-r)*items[i][1]); run=max(run,adj); out[i]=(items[i][0],items[i][1],run,run<0.05)
    return out
R={k:json.load(open(f"bspl_replay_PickCube-v1_k{k}.json")) for k in (2,4)}
print("### (A) PickCube-v1 open-loop paired replay, n=993 — QP vs bounded-error B-spline (Alg.1, eps=0.005)")
arms=R[2]["arms"]; print(f"{'arm':22s}{'k=2':>10s}{'k=4':>10s}")
for a in arms: print(f"{a:22s}"+"".join(f"{100*np.mean(R[k]['success'][a]):9.2f}%" for k in (2,4)))
print("\ncontrasts (delta pp, wins/losses, p, 95% CI):")
pairs=[("qp","bspline_eps_satfix"),("qp","cubic_spline_satfix"),("bspline_eps_satfix","exact_integral"),("bspline_eps_satfix","cubic_spline_satfix"),("bspline_eps_satfix","tac_fold_satfix")]
for k in (2,4):
    S=R[k]["success"]
    for a,b in pairs:
        c=R[k]["contrasts"].get(f"{a}_vs_{b}")
        if c: print(f"k={k} {a:20s} vs {b:20s} {c['delta_pp']:+6.2f}pp {c['wins']:3d}/{c['losses']:3d} p={c['p']:.3g} CI=[{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]")
        else:
            ao,bo,p=mcn(S[a],S[b]); print(f"k={k} {a:20s} vs {b:20s} {100*(np.mean(S[a])-np.mean(S[b])):+6.2f}pp {ao:3d}/{bo:3d} p={p:.3g}")
p1=[(f"k={k}",R[k]["contrasts"]["qp_vs_bspline_eps_satfix"]["p"]) for k in (2,4)]
print("\nP1 (primary) qp > bspline_eps_satfix at both k, Holm m=2:")
ok=True
for lbl,p,ph,s in holm(p1):
    k=int(lbl[2]); d=R[k]["contrasts"]["qp_vs_bspline_eps_satfix"]["delta_pp"]; ok&=(d>0 and s); print(f"  {lbl}: {d:+.2f}pp raw p={p:.3g} Holm p={ph:.3g} survives={s}")
print("  P1", "HOLDS" if ok else "FAILS")
for k in (2,4):
    S=R[k]["success"]; b=100*np.mean(S["bspline_eps_satfix"]); z=100*np.mean(S["exact_integral"]); c=100*np.mean(S["cubic_spline_satfix"])
    print(f"P2 k={k}: bspline {b:.2f} > ZOH {z:.2f}? {b>z} (p={mcn(S['bspline_eps_satfix'],S['exact_integral'])[2]:.3g})   P3 k={k}: bspline < cubic_satfix {c:.2f}? {b<c} (p={mcn(S['bspline_eps_satfix'],S['cubic_spline_satfix'])[2]:.3g})")
b2=100*np.mean(R[2]["success"]["bspline_eps_satfix"]); print(f"P4: bspline_eps_satfix@k=2 = {b2:.2f}% ; between ZOH 58 and cubic 72.7? {58<b2<72.7}")
print("\n### (B) RoboCasa OpenDrawer closed-loop DP, k=2, n=15 (new 4-arm run)")
r=json.load(open("result_dp_RC-OpenDrawer.json")); S=r["success"]
print("arms:", r["arms"], "| raw sat:", round(r["policy_raw_sat_frac"]["native"],4))
for a in r["arms"]: print(f"  {a:16s} {sum(S[a])}/15")
for ref in ("zoh","native","spline_satfix"):
    ao,bo,p=mcn(S["qp"],S[ref]); print(f"  qp vs {ref:14s} {100*(np.mean(S['qp'])-np.mean(S[ref])):+6.1f}pp {ao}/{bo} p={p:.3g}")
