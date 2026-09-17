import os, glob, json

def parse_detailed():
    base_dir = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
    
    # We want LIBERO-Spatial 10 tasks
    # Let's inspect spatial40_out, standard_flow_s0, standard_til_s0, r5g_flow30_spatial_native_20env_out, etc.
    
    files_to_check = {
        "DeepONet Native 20Hz": f"{base_dir}/spatial40_out/multirate_libero_spatial.json",
        "DeepONet 40Hz Naive": f"{base_dir}/spatial40_out/multirate_libero_spatial.json",
        "DeepONet 40Hz Spline": f"{base_dir}/spatial40_out/multirate_libero_spatial.json",
        "DeepONet 40Hz Folding": f"{base_dir}/spatial40_out/multirate_libero_spatial.json",
    }
    
    # Let's search all json files under spatial40_out and others
    for f in glob.glob(f"{base_dir}/spatial40_out/*.json"):
        print("spatial40 file:", f)
        with open(f) as fp:
            d = json.load(fp)
            for arm in d:
                print("  ARM:", arm)
                if isinstance(d[arm], dict) and "per_task" in d[arm]:
                    print("   Aggregate:", d[arm].get("aggregate"))
                    for tid, tinfo in d[arm]["per_task"].items():
                        print(f"     Task {tid} ({tinfo['task'][:35]}): {tinfo['success_rate']*100:.1f}%")

    print("\n--- Checking Flow Arms ---")
    for f in glob.glob(f"{base_dir}/r5g_flow30_spatial_native_20env_out/*.json") + glob.glob(f"{base_dir}/*flow*spatial*/*.json") + glob.glob(f"{base_dir}/matrix_eval_out/*.json"):
        print("Flow file:", f)
        with open(f) as fp:
            d = json.load(fp)
            if isinstance(d, dict):
                for arm in d:
                    print("  ARM:", arm)
                    if isinstance(d[arm], dict) and "per_task" in d[arm]:
                        print("   Aggregate:", d[arm].get("aggregate"))
                        for tid, tinfo in d[arm]["per_task"].items():
                            print(f"     Task {tid} ({tinfo['task'][:35]}): {tinfo['success_rate']*100:.1f}%")

if __name__ == "__main__":
    parse_detailed()
