import json, glob, os
from math import comb
BASE="/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"

def betainc(a,b,x):
    n=a+b-1
    return sum(comb(n,k)*x**k*(1-x)**(n-k) for k in range(a,n+1))
def bppf(p,a,b):
    lo,hi=0.0,1.0
    for _ in range(200):
        m=(lo+hi)/2
        if betainc(a,b,m)<p: lo=m
        else: hi=m
    return (lo+hi)/2
def cp(k,n,c=0.90):
    al=1-c
    return (0.0 if k==0 else bppf(al/2,k,n-k+1)), (1.0 if k==n else bppf(1-al/2,k+1,n-k))
def mcn(b,c):
    n=b+c
    if n==0: return 1.0
    k=min(b,c); return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)

def load(pat):
    fs=glob.glob(os.path.join(BASE,pat,"*.json"))
    if not fs: return None
    d=json.load(open(fs[0]))
    for k,v in d.items():
        if v.get("aggregate") is None: continue
        return {(t,e["seed"]):e["success"] for t,tv in v["per_task"].items() for e in tv["episodes"]}
    return None

print(f"{'suite':<9}{'native@20':>11}{'spline@40':>11}{'ours@40':>10}{'diff':>9}{'disc':>8}{'p':>9}{'90% CI':>18}  verdict")
print("-"*104)
for suite,m in (("object","object"),("goal","goal"),("long","long")):
    N=load(f"suiteeval_{m}_asrc_{m}_native_20env_out")
    S=load(f"suiteeval_{m}_asrc_{m}_cadmag_spline_40env_out") or load(f"suiteeval_long_asrc_{m}_cadmag_spline_40env_out")
    F=load(f"suiteeval_{m}_asrc_{m}_cadmag_folding_40env_out") or load(f"suiteeval_long_asrc_{m}_cadmag_folding_40env_out")
    if not (N and S and F): print(f"{suite:<9} MISSING (N={bool(N)} S={bool(S)} F={bool(F)})"); continue
    ks=sorted(set(S)&set(F)); n=len(ks)
    b=sum(1 for k in ks if F[k] and not S[k]); c=sum(1 for k in ks if S[k] and not F[k])
    d=(b-c)/n*100
    lo,hi=cp(b,b+c,0.90) if (b+c)>0 else (0.5,0.5)
    lo=(2*lo-1)*(b+c)/n*100; hi=(2*hi-1)*(b+c)/n*100
    p=mcn(b,c); marg=max(abs(lo),abs(hi))
    v="TIE (certified<=5pp)" if p>=0.05 and marg<=5 else ("TIE (undetermined)" if p>=0.05 else "DIFFERENT")
    na=sum(N.values())/len(N)*100
    print(f"{suite:<9}{na:>10.1f}%{sum(S[k] for k in ks)/n*100:>10.1f}%{sum(F[k] for k in ks)/n*100:>9.1f}%"
          f"{d:>+8.1f}pp{f'{b}/{c}':>8}{p:>9.4f}{f'[{lo:+.1f},{hi:+.1f}]':>18}  {v} n={n}")
