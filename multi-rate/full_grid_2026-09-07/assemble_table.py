"""Paper-style closed-loop table (Diffusion Policy): rows = resampler arm, cols = task, one table per k.
Loads result_dp_<task>.json (k=2) and result_dp_<task>_k4.json (k=4). '—' = arm/cell not run."""
import json, os, math, numpy as np
TASKS=[("PushT-v1","PushT"),("PickCube-v1","PickCube"),("lift","Lift"),("can","Can"),("square","Square"),("RC-OpenDrawer","RC OpenDrawer"),("RC-PnPCounterToStove","RC PnPStove")]
ROWS=[("native","Diff. 1X Base (native)"),("zoh","Diff. + ZOH"),("spline_satfix","Diff. + spline+satfix"),("tac_fold_satfix","Diff. + TAC-Fold+satfix"),("bspline_eps_satfix","Diff. + B-spline (Alg.1)+satfix"),("qp","Diff. + QP (ours)")]
def mcn(a,b):
    a,b=np.asarray(a,bool),np.asarray(b,bool); ao=int((a&~b).sum()); bo=int((~a&b).sum()); d=ao+bo
    return ao,bo,(1.0 if d==0 else float(min(1.0,2*sum(math.comb(d,i) for i in range(min(ao,bo)+1))/2**d)))
def load(task,k):
    f=f"result_dp_{task}.json" if k==2 else f"result_dp_{task}_k{k}.json"
    return json.load(open(f)) if os.path.exists(f) else None
for k in (2,4):
    R={t:load(t,k) for t,_ in TASKS}
    print(f"\n### Closed-loop success, Diffusion Policy, k={k}")
    hdr="| Configuration | "+" | ".join(f"{lbl} (n={R[t]['n']})" if R[t] else f"{lbl} (—)" for t,lbl in TASKS)+" |"
    print(hdr); print("|"+"---|"*(len(TASKS)+1))
    for arm,lbl in ROWS:
        cells=[]
        for t,_ in TASKS:
            r=R[t]; cells.append("—" if (r is None or arm not in r["success"]) else f"{100*np.mean(r['success'][arm]):.0f}%")
        print(f"| {lbl} | "+" | ".join(cells)+" |")
    print(f"\nQP paired exact McNemar, k={k} (Δpp, qp-only/ref-only, p):")
    for t,lbl in TASKS:
        r=R[t]
        if r is None or "qp" not in r["success"]: continue
        S=r["success"]; line=[]
        for ref in ("zoh","native","spline_satfix","tac_fold_satfix","bspline_eps_satfix"):
            if ref in S:
                ao,bo,p=mcn(S["qp"],S[ref]); line.append(f"vs {ref} {100*(np.mean(S['qp'])-np.mean(S[ref])):+.0f}pp {ao}/{bo} p={p:.2g}")
        print(f"  {lbl:14s} "+" | ".join(line))
