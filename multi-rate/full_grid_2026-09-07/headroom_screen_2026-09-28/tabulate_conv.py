#!/usr/bin/env python3
"""Amendment 3 tabulation: for each eligible family/k, is native - max(3 converters) >= 10pp on >= half the tasks?
Run after the converter-check jobs finish. Reads only; writes nothing."""
import json, glob, os

def pc(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    if "overall" in d: return d["overall"]["pc_success"]  # lerobot eval_info.json
    return 100 * sum(d["successes"]) / d["n"]  # eval_dp_official_rate.py / eval_pusht_hub_resamplers.py style

def verdict(family, rows):
    """rows: list of (task, native, {arm: pc})"""
    open_tasks, closed_tasks = [], []
    for task, native, conv in rows:
        if native is None or any(v is None for v in conv.values()):
            print(f"  {task:20s} native={native} conv={conv}  INCOMPLETE"); continue
        best_conv = max(conv.values())
        gap = native - best_conv
        tag = "OPEN " if (native >= 30 and gap >= 10) else "closed"
        (open_tasks if tag == "OPEN " else closed_tasks).append(task)
        print(f"  {task:20s} native={native:5.1f} best_conv={best_conv:5.1f} ({', '.join(f'{a}={v:.1f}' for a,v in conv.items())}) gap={gap:+5.1f}pp  {tag}")
    n = len(open_tasks) + len(closed_tasks)
    verdict = "CANDIDATE" if n and len(open_tasks) >= n / 2 else ("NOT ENOUGH DATA" if n == 0 else "converters close the gap")
    print(f"  => {family}: {len(open_tasks)}/{n} open  =>  {verdict}\n")

print("=== Push-T ===")
for k in (2, 4):
    f = f"pusht_conv_k{k}.json"
    if not os.path.exists(f): print(f"  k={k}: not done yet"); continue
    d = json.load(open(f))
    arms = d.get("arms", {})
    native = arms.get("native", {}).get("pc_success")
    if native is None and os.path.exists("pusht_k4_screen.json"):  # native (k=1) lives only in the k4 screen file
        native = json.load(open("pusht_k4_screen.json"))["arms"].get("native", {}).get("pc_success")
    conv = {a: arms.get(a, {}).get("pc_success") for a in ("spline_satfix", "tac_fold_satfix", "qp_anchor")}
    print(f" k={k}:")
    verdict("Push-T", [("PushT", native, conv)])

print("=== RoboMimic (k=4) ===")
rows = []
for t in ("can_ph", "square_ph", "transport_ph", "tool_hang_ph"):
    native = pc(f"dp_{t}_native_k1.json")
    conv = {a: pc(f"dp_{t}_{a}_k4.json") for a in ("spline_satfix", "tac_fold_satfix", "qp_anchor")}
    rows.append((t, native, conv))
verdict("RoboMimic", rows)

print("=== LIBERO (k=8) ===")
rows = []
for t in ("libero_10", "libero_object", "libero_goal", "libero_spatial"):
    native = pc(f"libero_{t}_native_k1/eval_info.json")
    conv = {a: pc(f"libero_{t}_{a}_k8/eval_info.json") for a in ("spline_satfix", "tac_fold_satfix", "qp_anchor")}
    rows.append((t, native, conv))
verdict("LIBERO", rows)

print("=== Franka Kitchen ===")
for k in (2, 4):
    native = pc("dp_kitchen_native_k1.json")
    conv = {a: pc(f"dp_kitchen_{a}_k{k}.json") for a in ("spline_satfix", "tac_fold_satfix", "qp_anchor")}
    print(f" k={k}:")
    verdict("Kitchen", [("kitchen", native, conv)])
