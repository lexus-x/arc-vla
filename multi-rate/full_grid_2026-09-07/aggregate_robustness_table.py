#!/usr/bin/env python3
"""Aggregate empirical robustness evaluation results into a clean markdown table."""
import glob, json, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def get_stats(task, k, noise_str):
    # Try finding the file
    path = f"{HERE}/result_dp_{task}_k{k}_{noise_str}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    succ = d.get("success", {})
    return succ

def format_rate(succ_list):
    if succ_list is None:
        return "Pending"
    s = sum(succ_list)
    n = len(succ_list)
    return f"{100.0 * s / n:.1f}% ({s}/{n})"

def main():
    tasks = ["lift", "can", "square"]
    ks = [1, 2, 4, 8]
    methods = [
        ("Diff. Base", lambda succ, k: succ.get("native") if k == 1 else succ.get("zoh")),
        ("Diff. + Spline", lambda succ, k: succ.get("spline_satfix")),
        ("Diff. + B-spline", lambda succ, k: succ.get("bspline_satfix")),
        ("Diff. + Ours (QP)", lambda succ, k: succ.get("qp")),
    ]

    print("# Empirical RoboMimic Robustness Scoreboard (State-Space Observation Noise $\\sigma=0.05$, $n=100$ per cell)\n")
    print("| Task | Speedup ($K$) | Perturbation (Noise $\\sigma$) | Diff. Base | Diff. + Spline | Diff. + B-spline | Diff. + Ours (QP) | Best Method | Margin (Ours vs Base / vs B-spline) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for task in tasks:
        for i, k in enumerate(ks):
            succ = get_stats(task, k, "noise05")
            row = []
            rates = {}
            for name, getter in methods:
                s_list = getter(succ, k) if succ else None
                rates[name] = s_list
                row.append(format_rate(s_list))

            # Compute best method and margin
            valid_rates = {m: (sum(vals)/len(vals)) for m, vals in rates.items() if vals is not None}
            if len(valid_rates) == len(methods):
                best_m = max(valid_rates, key=valid_rates.get)
                qp_r = valid_rates["Diff. + Ours (QP)"]
                base_r = valid_rates["Diff. Base"]
                bspl_r = valid_rates["Diff. + B-spline"]
                d_base = (qp_r - base_r) * 100
                d_bspl = (qp_r - bspl_r) * 100
                best_str = f"**{best_m}**" if best_m == "Diff. + Ours (QP)" else best_m
                margin_str = f"{d_base:+4.1f}pp / {d_bspl:+4.1f}pp"
            else:
                best_str = "Pending"
                margin_str = "Pending"

            task_col = f"**{task.capitalize()}**" if i == 0 else ""
            print(f"| {task_col:10s} | {k}X | $\\sigma=0.05$ (5% noise) | {row[0]:12s} | {row[1]:14s} | {row[2]:16s} | **{row[3]}** | {best_str} | {margin_str} |")

if __name__ == "__main__":
    main()
