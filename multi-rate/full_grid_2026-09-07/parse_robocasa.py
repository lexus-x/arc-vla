import json, os

base = '/home/user/Desktop/multi-rate/full_grid_2026-09-07'
tasks = ['CloseSingleDoor', 'CoffeePressButton', 'TurnOffMicrowave', 'TurnOffSinkFaucet']
ks = ['k1', 'k4', 'k8']

print(f"| {'Task':<20} | {'Rate':<5} | {'TAC-Fold':<10} | {'Spline':<10} | {'B-Spline':<10} | {'Winner':<10} |")
print("|" + "-"*22 + "|" + "-"*7 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")

for task in tasks:
    for k in ks:
        fn = os.path.join(base, f"result_dp_RC-{task}_f0_{k}_random_n100.json")
        if os.path.exists(fn):
            d = json.load(open(fn))
            succ = d.get('success', {})
            tf_list = succ.get('tac_fold_satfix', [False])
            sp_list = succ.get('spline', [False])
            bs_list = succ.get('bspline_eps_raw', [False])
            tf = (sum(tf_list) / len(tf_list)) * 100
            sp = (sum(sp_list) / len(sp_list)) * 100
            bs = (sum(bs_list) / len(bs_list)) * 100
            
            if tf > max(sp, bs):
                w = 'TAC-Fold'
            elif tf == max(sp, bs):
                w = 'Tie'
            elif sp > bs:
                w = 'Spline'
            else:
                w = 'B-Spline'
            print(f"| {task:<20} | {k:<5} | {tf:5.1f}%    | {sp:5.1f}%    | {bs:5.1f}%    | {w:<10} |")
