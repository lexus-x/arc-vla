import json, math, numpy as np
def mcnemar(a,b):
    a,b=np.asarray(a,bool),np.asarray(b,bool); ao=int((a&~b).sum()); bo=int((~a&b).sum()); d=ao+bo
    if d==0: return ao,bo,1.0
    return ao,bo,float(min(1.0,2*sum(math.comb(d,i) for i in range(min(ao,bo)+1))/2**d))
def holm(items):
    order=sorted(range(len(items)),key=lambda i:items[i][1]); m=len(items); out=[None]*m; run=0.0
    for r,i in enumerate(order):
        adj=min(1.0,(m-r)*items[i][1]); run=max(run,adj); out[i]=(items[i][0],items[i][1],run,run<0.05)
    return out
print("### (A) PickCube-v1 closed-loop Diffusion Policy, n=100, paired")
R={k:json.load(open(f"result_dp_PickCube-v1_k{k}.json")) for k in (4,5)}
arms=R[4]["arms"]; print(f"{'arm':18s}{'k=4':>10s}{'k=5':>10s}")
for a in arms: print(f"{a:18s}"+"".join(f"{100*np.mean(R[k]['success'][a]):9.1f}%" for k in (4,5)))
print("k=5 arms identical?", all(R[5]['success'][a]==R[5]['success']['zoh'] for a in ('spline_satfix','tac_fold_satfix','qp')))
fam=[]
for k in (4,5):
    S=R[k]["success"]
    for ref in ("zoh","native"):
        c=R[k]["contrasts"]["qp"][f"vs_{ref}"]; print(f"k={k} qp vs {ref:16s} {c['delta_pp']:+6.1f}pp  {c['a_only']:2d}/{c['ref_only']:2d}  p={c['p']:.3g}")
    for ref in ("spline_satfix","tac_fold_satfix"):
        ao,bo,p=mcnemar(S["qp"],S[ref]); d=100*(np.mean(S["qp"])-np.mean(S[ref]))
        print(f"k={k} qp vs {ref:16s} {d:+6.1f}pp  {ao:2d}/{bo:2d}  p={p:.3g}"); fam.append((f"k={k} qp_vs_{ref}",p))
print("Holm m=4 over {qp vs spline_satfix, qp vs tac_fold_satfix} x {k4,k5}:")
for lbl,p,ph,s in holm(fam): print(f"  {lbl:28s} raw p={p:.3g} Holm p={ph:.3g} {'SURVIVES' if s else ''}")
print("\n### (B) RoboCasa OpenDrawer closed-loop DP, k=4, n=15")
r=json.load(open("result_dp_RC-OpenDrawer_k4.json"))
for a in r["arms"]: print(f"{a:18s} {100*np.mean(r['success'][a]):6.1f}%  ({sum(r['success'][a])}/15)")
print("policy raw |a|>1 frac:", round(r["policy_raw_sat_frac"]["native"],4))
for ref in ("zoh","native"):
    c=r["contrasts"]["qp"][f"vs_{ref}"]; print(f"qp vs {ref:8s} {c['delta_pp']:+6.1f}pp {c['a_only']}/{c['ref_only']} p={c['p']:.3g}")
ao,bo,p=mcnemar(r["success"]["qp"],r["success"]["spline_satfix"]); print(f"qp vs spline_satfix {100*(np.mean(r['success']['qp'])-np.mean(r['success']['spline_satfix'])):+6.1f}pp {ao}/{bo} p={p:.3g}")
