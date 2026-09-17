import json, os
B = "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH"
rows = []
for S in ("Spatial", "Object", "Long"):
    for tag, p in (("flow", f"{B}/paper_repro/{S}/runs/eval_flow"),
                   ("v2", f"{B}/v2/deeponet_results/{S}/runs/eval_m3")):
        try:
            avg = json.load(open(f"{p}/summary_full.json")).get("average_all_tasks_pct")
        except Exception as e:
            avg = f"ERR {e}"
        proto = ""
        try:
            c = json.load(open(f"{p}/success_rates.json"))["_config"]
            k = list(c)[0]
            proto = {x: c[k][x] for x in ("replan", "max_steps", "indist_episodes") if x in c[k]}
        except Exception as e:
            proto = f"ERR {e}"
        rows.append((S, tag, avg, proto))
for S, tag, avg, proto in rows:
    print(f"{S:8s} {tag:5s} {str(avg):>6}   {proto}")
