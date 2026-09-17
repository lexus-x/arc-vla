import json, glob, os
from math import comb
BASE="/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
def mcn(b,c):
    n=b+c
    if n==0: return 1.0
    k=min(b,c); return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)
# gather every til arm anywhere on disk
arms={}
for f in glob.glob(os.path.join(BASE,"*","*.json")):
    try: d=json.load(open(f))
    except Exception: continue
    for k,v in d.items():
        if not k.startswith("til") or not isinstance(v,dict): continue
        if v.get("aggregate") is None or "per_task" not in v: continue
        try:
            ep={(t,e["seed"]):e["success"] for t,tv in v["per_task"].items() for e in tv["episodes"]}
        except Exception: continue
        if k not in arms or len(ep)>len(arms[k]): arms[k]=ep
print("til arms found:", len(arms))
for k in sorted(arms): print(f"   {k:<34} {sum(arms[k].values())}/{len(arms[k])} = {sum(arms[k].values())/len(arms[k])*100:.1f}%")
print()
for rate in (30,40):
    F=arms.get(f"til_cadmag_folding_{rate}env"); S=arms.get(f"til_cadmag_spline_{rate}env")
    if not (F and S): print(f"@{rate}Hz: missing paired data"); continue
    ks=sorted(set(F)&set(S)); n=len(ks)
    b=sum(1 for k in ks if F[k] and not S[k]); c=sum(1 for k in ks if S[k] and not F[k])
    print(f"til folding vs spline @{rate}Hz: n={n}  {(b-c)/n*100:+.1f}pp  disc {b}/{c}  p={mcn(b,c):.4f}")
