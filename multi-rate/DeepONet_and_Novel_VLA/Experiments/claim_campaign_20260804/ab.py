import json, statistics as st
P = "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/Ablation_Results/ablation_robustness.json"
d = json.load(open(P))
print("config:", json.dumps(d.get("_config"), indent=1)[:400])
print()
arms = [k for k in d if k != "_config"]
print("arms:", arms)
print()
for a in arms:
    v = d[a]
    print(f"  {a:22s} {type(v).__name__} keys={list(v)[:6] if isinstance(v,dict) else len(v)}")
