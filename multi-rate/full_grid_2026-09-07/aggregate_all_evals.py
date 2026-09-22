import json
import os
import glob
import numpy as np

HERE = "/home/user/Desktop/multi-rate/full_grid_2026-09-07"

# Checkpoint paths and details
CHECKPOINTS = {
    "RoboMimic Lift": {
        "path": os.path.join(HERE, "dp_lift.pt"),
        "sha256": "893c5d6e27943d63ba424687d4aa3302bc6f3df89fc8c459f6b55ae2154388ff",
        "demos": "demos_lift_200.npz (200 demos)",
        "protocol": "heldout_demo_state (RoboMimic DP)",
        "n_eval": 50,
        "policy": "Diffusion Policy (RoboMimic)"
    },
    "RoboMimic Can": {
        "path": os.path.join(HERE, "dp_can.pt"),
        "sha256": "4bcf081bca6e0018fa1f4c718d7f4749f7e596bb5d0032c525a7a5be893e488b",
        "demos": "demos_can_200.npz (200 demos)",
        "protocol": "heldout_demo_state (RoboMimic DP)",
        "n_eval": 50,
        "policy": "Diffusion Policy (RoboMimic)"
    },
    "RoboMimic Square": {
        "path": os.path.join(HERE, "dp_square.pt"),
        "sha256": "3e7e4b32896701ddd3e121c1525419a536258c685b312639bb6df63583aeef98",
        "demos": "demos_square_200.npz (200 demos)",
        "protocol": "heldout_demo_state (RoboMimic DP)",
        "n_eval": 50,
        "policy": "Diffusion Policy (RoboMimic)"
    },
    "Push-T": {
        "path": "lerobot/diffusion_pusht (HuggingFace Hub / local cache)",
        "sha256": "Official LeRobot pretrained Diffusion Policy checkpoint",
        "demos": "pusht dataset (206 demos)",
        "protocol": "Seeded environment rollouts (max 300 steps, success threshold 95% coverage)",
        "n_eval": 50,
        "policy": "Diffusion Policy (LeRobot)"
    }
}

def load_robomimic_task(task, k_val):
    # k_val: 1, 2, 4, 8
    suffix = f"result_dp_{task}_k{k_val}_arc_eval.json" if k_val != 2 else f"result_dp_{task}_arc_eval.json"
    p = os.path.join(HERE, suffix)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        data = json.load(f)
    res = {}
    for arm in ["arc", "spline", "bspline_eps_raw"]:
        if arm in data.get("success", {}):
            succ = data["success"][arm]
            res[arm] = (sum(succ) / len(succ)) * 100.0
    return res

def load_all():
    print("=" * 80)
    print("GENUINE EVALUATION RESULTS: ROBOMIMIC & PUSH-T")
    print("=" * 80)

    # Rates to report: 20 Hz, 10 Hz, 5 Hz, 2.5 Hz, Native
    rates = ["20 Hz", "10 Hz", "5 Hz", "2.5 Hz", "Native"]
    
    # 1. RoboMimic per-task
    robomimic_tasks = ["lift", "can", "square"]
    # k mapping:
    # 20 Hz -> k=1
    # 10 Hz -> k=2
    # 5 Hz  -> k=4
    # 2.5 Hz -> k=8
    # Native -> k=1
    k_map = {
        "20 Hz": 1,
        "10 Hz": 2,
        "5 Hz": 4,
        "2.5 Hz": 8,
        "Native": 1,
    }

    robomimic_results = {t: {} for t in robomimic_tasks}
    for t in robomimic_tasks:
        for r in rates:
            k = k_map[r]
            res = load_robomimic_task(t, k)
            robomimic_results[t][r] = res

    # 2. Push-T
    pusht_file = os.path.join(HERE, "eval_arc_results.json")
    pusht_data = {}
    if os.path.exists(pusht_file):
        with open(pusht_file, "r") as f:
            pusht_data = json.load(f)

    # Merge eval_pusht_10hz.json and eval_pusht_20hz.json if present
    for extra_file in ["eval_pusht_10hz.json", "eval_pusht_20hz.json"]:
        p = os.path.join(HERE, extra_file)
        if os.path.exists(p):
            with open(p, "r") as f:
                extra_data = json.load(f)
                pusht_data.update(extra_data)
    
    # Save combined
    if pusht_data:
        with open(pusht_file, "w") as f:
            json.dump(pusht_data, f, indent=2)

    # pusht mapping:
    # 20 Hz -> 20.0Hz
    # 10 Hz -> 10.0Hz
    # 5 Hz  -> 5.0Hz
    # 2.5 Hz -> 2.5Hz
    # Native -> 10.0Hz (Native Push-T is 10 Hz)
    pusht_rate_map = {
        "20 Hz": "20.0Hz",
        "10 Hz": "10.0Hz",
        "5 Hz": "5.0Hz",
        "2.5 Hz": "2.5Hz",
        "Native": "10.0Hz",
    }
    
    pusht_results = {}
    for r in rates:
        pr_key = pusht_rate_map[r]
        if pr_key in pusht_data:
            pusht_results[r] = {
                arm: pusht_data[pr_key][arm]["pc_success"]
                for arm in ["arc", "spline", "bspline_eps_raw"]
                if arm in pusht_data[pr_key]
            }
        else:
            pusht_results[r] = None

    print("\n" + "=" * 80)
    print("PROVENANCE DETAILS")
    print("=" * 80)
    for name, info in CHECKPOINTS.items():
        print(f"[{name}]")
        print(f"  Path:     {info['path']}")
        print(f"  SHA-256:  {info['sha256']}")
        print(f"  Demos:    {info['demos']}")
        print(f"  Protocol: {info['protocol']} (n={info['n_eval']} episodes per condition)")
        print()

    print("=" * 80)
    print("BENCHMARK 1: ROBOMIMIC (PER-TASK)")
    print("=" * 80)
    for t in robomimic_tasks:
        print(f"\n--- RoboMimic: {t.capitalize()} ---")
        print(f"{'Rate':<10} | {'Condition':<10} | {'ARC (%)':<10} | {'Spline (%)':<12} | {'B-Spline (%)':<14} | {'Margin (ARC - Spline)':<22}")
        print("-" * 86)
        for r in rates:
            k = k_map[r]
            cond_str = f"k={k}" if r != "Native" else "k=1 (native)"
            res = robomimic_results[t][r]
            if res is not None and "arc" in res:
                arc_v = res["arc"]
                spl_v = res["spline"]
                bsp_v = res["bspline_eps_raw"]
                diff = arc_v - spl_v
                diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
                print(f"{r:<10} | {cond_str:<10} | {arc_v:<10.1f} | {spl_v:<12.1f} | {bsp_v:<14.1f} | {diff_str:<22}")
            else:
                print(f"{r:<10} | {cond_str:<10} | (running...)")

    # RoboMimic Average Table
    print("\n" + "=" * 80)
    print("ROBOMIMIC BENCHMARK AVERAGE (Mean across Lift, Can, Square)")
    print("=" * 80)
    print(f"{'Rate':<10} | {'Condition':<10} | {'ARC (%)':<10} | {'Spline (%)':<12} | {'B-Spline (%)':<14} | {'Margin (ARC - Spline)':<22}")
    print("-" * 86)
    rm_rate_avgs = {}
    for r in rates:
        k = k_map[r]
        cond_str = f"k={k}" if r != "Native" else "k=1 (native)"
        arc_list, spl_list, bsp_list = [], [], []
        for t in robomimic_tasks:
            res = robomimic_results[t][r]
            if res and "arc" in res:
                arc_list.append(res["arc"])
                spl_list.append(res["spline"])
                bsp_list.append(res["bspline_eps_raw"])
        if len(arc_list) == len(robomimic_tasks):
            m_arc = np.mean(arc_list)
            m_spl = np.mean(spl_list)
            m_bsp = np.mean(bsp_list)
            rm_rate_avgs[r] = (m_arc, m_spl, m_bsp)
            diff = m_arc - m_spl
            diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
            print(f"{r:<10} | {cond_str:<10} | {m_arc:<10.1f} | {m_spl:<12.1f} | {m_bsp:<14.1f} | {diff_str:<22}")
    
    # Overall RoboMimic Mean
    all_rm_arc = [rm_rate_avgs[r][0] for r in rates if r != "Native"]
    all_rm_spl = [rm_rate_avgs[r][1] for r in rates if r != "Native"]
    all_rm_bsp = [rm_rate_avgs[r][2] for r in rates if r != "Native"]
    print("-" * 86)
    diff_ov = np.mean(all_rm_arc) - np.mean(all_rm_spl)
    diff_ov_str = f"+{diff_ov:.1f}%" if diff_ov > 0 else f"{diff_ov:.1f}%"
    print(f"{'RM Overall':<10} | {'Mean (4 rates)':<10} | {np.mean(all_rm_arc):<10.1f} | {np.mean(all_rm_spl):<12.1f} | {np.mean(all_rm_bsp):<14.1f} | {diff_ov_str:<22}")

    # Push-T Table
    print("\n" + "=" * 80)
    print("BENCHMARK 2: PUSH-T")
    print("=" * 80)
    print(f"{'Rate':<10} | {'Condition':<10} | {'ARC (%)':<10} | {'Spline (%)':<12} | {'B-Spline (%)':<14} | {'Margin (ARC - Spline)':<22}")
    print("-" * 86)
    pusht_factor_str = {
        "20 Hz": "2x upsample",
        "10 Hz": "1x native",
        "5 Hz": "k=2",
        "2.5 Hz": "k=4",
        "Native": "1x native",
    }
    pt_rate_avgs = {}
    for r in rates:
        fstr = pusht_factor_str[r]
        res = pusht_results[r]
        if res is not None and "arc" in res:
            arc_v = res["arc"]
            spl_v = res["spline"]
            bsp_v = res["bspline_eps_raw"]
            pt_rate_avgs[r] = (arc_v, spl_v, bsp_v)
            diff = arc_v - spl_v
            diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
            print(f"{r:<10} | {fstr:<10} | {arc_v:<10.1f} | {spl_v:<12.1f} | {bsp_v:<14.1f} | {diff_str:<22}")
        else:
            print(f"{r:<10} | {fstr:<10} | (running...)")

    if len(pt_rate_avgs) == len(rates):
        all_pt_arc = [pt_rate_avgs[r][0] for r in rates if r != "Native"]
        all_pt_spl = [pt_rate_avgs[r][1] for r in rates if r != "Native"]
        all_pt_bsp = [pt_rate_avgs[r][2] for r in rates if r != "Native"]
        print("-" * 86)
        diff_pt = np.mean(all_pt_arc) - np.mean(all_pt_spl)
        diff_pt_str = f"+{diff_pt:.1f}%" if diff_pt > 0 else f"{diff_pt:.1f}%"
        print(f"{'PT Overall':<10} | {'Mean (4 rates)':<10} | {np.mean(all_pt_arc):<10.1f} | {np.mean(all_pt_spl):<12.1f} | {np.mean(all_pt_bsp):<14.1f} | {diff_pt_str:<22}")

    # Grand Average across benchmarks
    if len(rm_rate_avgs) == len(rates) and len(pt_rate_avgs) == len(rates):
        print("\n" + "=" * 80)
        print("GRAND AVERAGE (Mean across RoboMimic and Push-T)")
        print("=" * 80)
        print(f"{'Rate':<10} | {'RoboMimic Condition':<20} | {'Push-T Condition':<18} | {'ARC (%)':<10} | {'Spline (%)':<12} | {'B-Spline (%)':<14} | {'Margin (ARC - Spline)':<22}")
        print("-" * 115)
        grand_arc, grand_spl, grand_bsp = [], [], []
        for r in rates:
            rm_a, rm_s, rm_b = rm_rate_avgs[r]
            pt_a, pt_s, pt_b = pt_rate_avgs[r]
            g_a = (rm_a + pt_a) / 2.0
            g_s = (rm_s + pt_s) / 2.0
            g_b = (rm_b + pt_b) / 2.0
            if r != "Native":
                grand_arc.append(g_a)
                grand_spl.append(g_s)
                grand_bsp.append(g_b)
            diff = g_a - g_s
            diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
            rm_cond = f"k={k_map[r]}" if r != "Native" else "k=1 (native 20 Hz)"
            pt_cond = pusht_factor_str[r]
            print(f"{r:<10} | {rm_cond:<20} | {pt_cond:<18} | {g_a:<10.1f} | {g_s:<12.1f} | {g_b:<14.1f} | {diff_str:<22}")
        print("-" * 115)
        tot_a = np.mean(grand_arc)
        tot_s = np.mean(grand_spl)
        tot_b = np.mean(grand_bsp)
        diff_tot = tot_a - tot_s
        diff_tot_str = f"+{diff_tot:.1f}%" if diff_tot > 0 else f"{diff_tot:.1f}%"
        print(f"{'OVERALL':<10} | {'All 4 rates combined':<20} | {'All 4 rates combined':<18} | {tot_a:<10.1f} | {tot_s:<12.1f} | {tot_b:<14.1f} | {diff_tot_str:<22}")

if __name__ == "__main__":
    load_all()
