import json, glob, sys

files = sorted(glob.glob("multirate_honest_out/*.json"))
if not files:
    print("no files")
    sys.exit(0)
f = files[-1]
d = json.load(open(f))
print("FILE:", f)
print("ARMS:", list(d.keys())[:8])
k = list(d.keys())[0]
print("arm keys:", list(d[k].keys()))
print("per_task count:", len(d[k]["per_task"]))
t0 = list(d[k]["per_task"].values())[0]
print("task entry keys:", list(t0.keys()))
print("episode sample:", t0["episodes"][0])
print("config sample:", d[k]["config"])
