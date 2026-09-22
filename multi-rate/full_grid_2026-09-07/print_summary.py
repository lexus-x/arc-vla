import json, os, glob

tasks = ['lift', 'can', 'square']
rates = [
    ('20 Hz', 1),
    ('10 Hz', 2),
    ('5 Hz', 4),
    ('2.5 Hz', 8),
    ('Native', 1),
]

robomimic_data = {}

for t in tasks:
    robomimic_data[t] = {}
    for r_name, k in rates:
        arc_f = f'result_dp_{t}_k{k}_arc_eval.json' if k != 2 else f'result_dp_{t}_arc_eval.json'
        qp_f = f'result_dp_{t}_k{k}_qp_eval.json' if k != 2 else f'result_dp_{t}_qp_eval.json'
        
        with open(arc_f) as f:
            d_arc = json.load(f)
        with open(qp_f) as f:
            d_qp = json.load(f)
            
        res = {}
        for arm in ['arc', 'spline', 'bspline_eps_raw']:
            s = d_arc['success'][arm]
            res[arm] = (sum(s), len(s), 100.0 * sum(s) / len(s))
        for arm in ['tac_fold_satfix', 'qp_anchor']:
            s = d_qp['success'][arm]
            res[arm] = (sum(s), len(s), 100.0 * sum(s) / len(s))
        robomimic_data[t][r_name] = res

print('=== ROBOMIMIC DATA LOADED ===')
for t in tasks:
    print(f'\nTask: {t}')
    for r_name, k in rates:
        row = robomimic_data[t][r_name]
        print(f"  {r_name:<8} | ARC: {row['arc'][2]:5.1f}% | Spline: {row['spline'][2]:5.1f}% | B-Spline: {row['bspline_eps_raw'][2]:5.1f}% | TAC-Fold+Satfix: {row['tac_fold_satfix'][2]:5.1f}% | QP-Anchor: {row['qp_anchor'][2]:5.1f}%")

with open('summary_all_metrics.json', 'w') as f:
    json.dump(robomimic_data, f, indent=2)
print('Saved summary_all_metrics.json')
