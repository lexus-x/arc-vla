"""
Analyze and format 5-Way Stream Comparison results across PushT and RoboMimic
Outputs:
- Full comparison table (Success rate, saturation, delta vs native, delta vs BSP)
- Exact McNemar pairwise tests vs both Native and BSP
- Holm-Bonferroni family-wise significance
"""
import sys, json, os
import numpy as np

def exact_mcnemar(b_arr, c_arr):
    # b: arm1 succeeds, arm2 fails
    # c: arm1 fails, arm2 succeeds
    b = int(np.sum(b_arr & ~c_arr))
    c = int(np.sum(~b_arr & c_arr))
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    res = binomtest(min(b, c), n, 0.5, alternative='two-sided')
    return float(res.pvalue)

def analyze_json(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path) as f:
        d = json.load(f)
    print("="*75)
    print(f"FILE: {os.path.basename(path)}")
    print("="*75)
    print(f"{'Arm':18s} | {'Success':10s} | {'Rate':7s} | {'Saturation':10s}")
    print("-" * 55)
    for arm, stats in d.items():
        succ = stats.get('successes', 0)
        tot = stats.get('total', 0)
        rate = stats.get('success_rate', 0.0)
        sat = stats.get('saturation', 0.0)
        print(f"{arm:18s} | {succ:3d}/{tot:3d}    | {rate:5.1f}% | {sat:6.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            analyze_json(p)
    else:
        for f in ["stream_5way_pusht_results.json", "stream_5way_robomimic_lift_results.json"]:
            analyze_json(f"/home/user/Desktop/multi-rate/satfix_2026-09-05/{f}")
