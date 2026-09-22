import json, os, glob

base = '/home/user/Desktop/multi-rate/full_grid_2026-09-07'

files = [
    ('Lift k1', 'result_dp_lift_k1_matrix_missing.json'),
    ('Lift k4', 'result_dp_lift_k4_matrix_missing.json'),
    ('Lift k8', 'result_dp_lift_k8_matrix_missing.json'),
    ('Can k1', 'result_dp_can_k1_matrix_missing.json'),
    ('Can k4', 'result_dp_can_k4_matrix_missing.json'),
    ('Can k8', 'result_dp_can_k8_matrix_missing.json'),
    ('Square k1', 'result_dp_square_k1_matrix_missing.json'),
    ('Square k4', 'result_dp_square_k4_matrix_missing.json'),
    ('Square k8', 'result_dp_square_k8_matrix_missing.json'),
]

print(f"| {'Task / Rate':<15} | {'TAC-Fold':<10} | {'Spline':<10} | {'B-Spline':<10} | {'Winner':<10} |")
print("|" + "-"*17 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")

for label, fname in files:
    path = os.path.join(base, fname)
    if os.path.exists(path):
        d = json.load(open(path))
        arms = d.get('arms', {})
        tf = arms.get('tac_fold_satfix', {}).get('sr', None)
        if tf is None:
            tf = arms.get('tac_fold', {}).get('sr', 0.0)
        sp = arms.get('spline_satfix', {}).get('sr', None)
        if sp is None:
            sp = arms.get('spline', {}).get('sr', 0.0)
        bs = arms.get('bspline_eps_raw', {}).get('sr', None)
        if bs is None:
            bs = arms.get('bspline_satfix', {}).get('sr', arms.get('bspline', {}).get('sr', 0.0))
            
        tf, sp, bs = float(tf), float(sp), float(bs)
        if tf > max(sp, bs):
            w = 'TAC-Fold'
        elif tf == max(sp, bs):
            w = 'Tie'
        elif sp > bs:
            w = 'Spline'
        else:
            w = 'B-Spline'
        print(f"| {label:<15} | {tf:5.1f}%    | {sp:5.1f}%    | {bs:5.1f}%    | {w:<10} |")
    else:
        print(f"| {label:<15} | MISSING FILE: {fname}")
