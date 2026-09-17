import json
from math import comb
BASE="/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
exec(open(f"{BASE}/final_results.py").read().split('S = spatial(')[0])

def sp(p):
    d=json.load(open(p))
    return {k:{(t,e["seed"]):e["success"] for t,tv in v["per_task"].items() for e in tv["episodes"]}
            for k,v in d.items() if v.get("aggregate") is not None}
def pl(p):
    d=json.load(open(p))
    return {k:{t:e["success"] for t,e in v["per_task"].items()}
            for k,v in d.items() if v.get("aggregate") is not None}

S40=sp(f"{BASE}/spatial40_out/multirate_honest.json"); S30=sp(f"{BASE}/powered_out/multirate_honest.json")
P40=pl(f"{BASE}/plus40_out/plus_multirate.json");      P30=pl(f"{BASE}/plus_multirate_out/plus_multirate.json")

CELLS=[("LIBERO-Spatial","30 Hz",S30.get("asrc_cadmag_folding_30env"),S30.get("asrc_cadmag_spline_30env")),
       ("LIBERO-Spatial","40 Hz",S40.get("asrc_cadmag_folding_40env"),S40.get("asrc_cadmag_spline_40env")),
       ("LIBERO-Plus",   "30 Hz",P30.get("asrc_cadmag_folding_30env"),P30.get("asrc_cadmag_spline_30env")),
       ("LIBERO-Plus",   "40 Hz",P40.get("asrc_cadmag_folding_40env"),P40.get("asrc_cadmag_spline_40env"))]

print(f"{'suite':<16}{'rate':<7}{'ours':>10}{'spline':>10}{'diff':>9}{'disc':>9}{'p':>9}{'90% CI':>18}  verdict")
print("-"*104)
for suite,rate,A,B in CELLS:
    if A is None or B is None: print(f"{suite:<16}{rate:<7}  MISSING"); continue
    ks=sorted(set(A)&set(B)); n=len(ks)
    b=sum(1 for k in ks if A[k] and not B[k]); c=sum(1 for k in ks if B[k] and not A[k])
    d,lo,hi,p,m=stats(b,c,n)
    oa=sum(A[k] for k in ks)/n*100; sa=sum(B[k] for k in ks)/n*100
    v = "TIE (certified <=5pp)" if (p>=0.05 and m<=5.0) else ("TIE (undetermined)" if p>=0.05 else "DIFFERENT")
    print(f"{suite:<16}{rate:<7}{oa:>9.1f}%{sa:>9.1f}%{d:>+8.1f}pp{f'{b}/{c}':>9}{p:>9.4f}{f'[{lo:+.1f},{hi:+.1f}]':>18}  {v}  (n={n})")
