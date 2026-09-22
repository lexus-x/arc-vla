import json, os, glob

base = "/home/user/Desktop/multi-rate/full_grid_2026-09-07"

print("="*80)
print("PUSH-T CLEAN TRI EVAL (n=50 episodes, satfix)")
print("="*80)
pusht_file = os.path.join(base, "clean_tri_eval_satfix_results.json")
if os.path.exists(pusht_file):
    data = json.load(open(pusht_file))
    print(f"| {'Rate':<8} | {'TAC-Fold':<18} | {'Spline':<18} | {'B-Spline':<18} | {'Winner':<10} |")
    print("|" + "-"*10 + "|" + "-"*20 + "|" + "-"*20 + "|" + "-"*20 + "|" + "-"*12 + "|")
    for rate in ["2.5Hz", "5.0Hz", "10.0Hz", "20.0Hz", "50.0Hz"]:
        arms = data.get(rate, {})
        tf = arms.get("tac_fold", {})
        sp = arms.get("spline", {})
        bs = arms.get("bspline_eps_raw", {})
        tf_sr = tf.get("pc_success", 0)
        sp_sr = sp.get("pc_success", 0)
        bs_sr = bs.get("pc_success", 0)
        tf_rew = tf.get("avg_max_reward", 0)
        sp_rew = sp.get("avg_max_reward", 0)
        bs_rew = bs.get("avg_max_reward", 0)
        
        if tf_sr > max(sp_sr, bs_sr):
            winner = "TAC-Fold"
        elif tf_sr == max(sp_sr, bs_sr):
            winner = "Tie"
        elif sp_sr > bs_sr:
            winner = "Spline"
        else:
            winner = "B-Spline"
            
        print(f"| {rate:<8} | {tf_sr:4.1f}% (r={tf_rew:.3f}) | {sp_sr:4.1f}% (r={sp_rew:.3f}) | {bs_sr:4.1f}% (r={bs_rew:.3f}) | {winner:<10} |")

print("\n" + "="*80)
print("ROBOCASA RESULTS (4 Tasks, n=100 random)")
print("="*80)
rc_tasks = ['CloseSingleDoor', 'CoffeePressButton', 'TurnOffMicrowave', 'TurnOffSinkFaucet']
for task in rc_tasks:
    print(f"\n--- RoboCasa: {task} ---")
    for kk in ["k1", "k4", "k8"]:
        fn = os.path.join(base, f"result_dp_RC-{task}_f0_{kk}_random_n100.json")
        if os.path.exists(fn):
            d = json.load(open(fn))
            arms = d.get("arms", [])
            print(f"  Rate {kk}:")
            if isinstance(arms, list):
                for item in arms:
                    name = item.get("name", item.get("arm", "unknown"))
                    sr = item.get("sr", item.get("success_rate", "N/A"))
                    print(f"    - {name:25s}: {sr}%")
            elif isinstance(arms, dict):
                for arm, v in arms.items():
                    sr = v.get("sr", v.get("success_rate", "N/A"))
                    print(f"    - {arm:25s}: {sr}%")
            elif "success_rate" in d:
                print(f"    Overall SR: {d['success_rate']}")
            else:
                print(f"    Keys: {list(d.keys())}")
