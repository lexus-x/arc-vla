import json, glob

def get_table():
    base = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
    
    # 1. DeepONet powered spatial 40hz
    with open(f"{base}/spatial40_out/multirate_honest.json") as f:
        spatial_data = json.load(f)
        
    print("=== DEEPONET ARMS ===")
    for arm in spatial_data:
        print(f"ARM: {arm} (Aggregate: {spatial_data[arm]['aggregate']*100:.1f}%)")
        for t, info in spatial_data[arm]["per_task"].items():
            print(f"  Task {t} [{info['task']}]: {info['success_rate']*100:.1f}% ({sum(e['success'] for e in info['episodes'])}/{len(info['episodes'])})")
                
    # 2. Flow Arms
    print("\n=== FLOW ARMS ===")
    with open(f"{base}/matrix_eval_out/matrix_comparison_40hz.json") as f:
        m40 = json.load(f)
    with open(f"{base}/matrix_eval_out/matrix_comparison_20hz.json") as f:
        m20 = json.load(f)
        
    for arm in ["flow_s0_unmodified_40hz", "flow_s0_spline_40hz", "flow_s0_folding_40hz"]:
        if arm in m40:
            print(f"ARM 40Hz: {arm} (Aggregate: {m40[arm]['aggregate']*100:.1f}%)")
            for t, info in m40[arm]["per_task"].items():
                print(f"  Task {t} [{info['task']}]: {info['success_rate']*100:.1f}%")
                
    for arm in ["flow_s0_spline_20hz", "flow_s0_folding_20hz"]:
        if arm in m20:
            print(f"ARM 20Hz: {arm} (Aggregate: {m20[arm]['aggregate']*100:.1f}%)")
            for t, info in m20[arm]["per_task"].items():
                print(f"  Task {t} [{info['task']}]: {info['success_rate']*100:.1f}%")

if __name__ == "__main__":
    get_table()
