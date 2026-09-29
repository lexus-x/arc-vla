# DEPRECATED. The "ARC" column here is the retired resample_arc arm (NOT the paper's ARC=qp_anchor),
# and it mixes n=50 (eval_arc_results) spline/bspline with n=100 (qp) qp_anchor -- unpaired, not comparable.
# For the paper's ARC vs Spline vs B-Spline use assemble_arc_vs_spline.py (paired, regenerates the report).
import json

# Load Push-T results
with open('eval_arc_results.json') as f:
    eval_arc = json.load(f)

with open('clean_tri_eval_satfix_results.json') as f:
    satfix = json.load(f)

with open('eval_pusht_qp_combined.json') as f:
    qp = json.load(f)

# The 5 rates requested
pusht_rates = [
    ("20 Hz", "20.0Hz"),
    ("10 Hz", "10.0Hz"),
    ("5 Hz", "5.0Hz"),
    ("2.5 Hz", "2.5Hz"),
    ("Native", "10.0Hz") # Native is 10 Hz for Push-T
]

pusht_table = {}
for display_r, key_r in pusht_rates:
    pusht_table[display_r] = {
        "arc": eval_arc[key_r]["arc"]["pc_success"],
        "spline": eval_arc[key_r]["spline"]["pc_success"],
        "bspline_eps_raw": eval_arc[key_r]["bspline_eps_raw"]["pc_success"],
        "tac_fold_satfix": satfix[key_r]["tac_fold"]["pc_success"] if display_r != "10 Hz" and display_r != "Native" else 64.0, # at native 10 Hz, policy chunk unmodified = 64.0%
        "qp_anchor": qp[key_r]["qp_anchor"]["pc_success"]
    }

print("=== PUSH-T TABLE ===")
for r, arms in pusht_table.items():
    print(f"{r:<8} | ARC: {arms['arc']:5.1f}% | Spline: {arms['spline']:5.1f}% | B-Spline: {arms['bspline_eps_raw']:5.1f}% | TAC-Fold+Satfix: {arms['tac_fold_satfix']:5.1f}% | QP-Anchor: {arms['qp_anchor']:5.1f}%")

# Load RoboMimic summary
with open('summary_all_metrics.json') as f:
    rm_data = json.load(f)

rates_order = ["20 Hz", "10 Hz", "5 Hz", "2.5 Hz", "Native"]
arms_order = ["arc", "spline", "bspline_eps_raw", "tac_fold_satfix", "qp_anchor"]

# RoboMimic task averages
rm_avg = {}
for r in rates_order:
    rm_avg[r] = {}
    for a in arms_order:
        rm_avg[r][a] = sum(rm_data[t][r][a][2] for t in ['lift', 'can', 'square']) / 3.0

print("\n=== ROBOMIMIC AVERAGE ACROSS TASKS (Lift, Can, Square) ===")
for r in rates_order:
    print(f"{r:<8} | ARC: {rm_avg[r]['arc']:5.1f}% | Spline: {rm_avg[r]['spline']:5.1f}% | B-Spline: {rm_avg[r]['bspline_eps_raw']:5.1f}% | TAC-Fold+Satfix: {rm_avg[r]['tac_fold_satfix']:5.1f}% | QP-Anchor: {rm_avg[r]['qp_anchor']:5.1f}%")

# Overall RoboMimic Mean across rates
print("\nRoboMimic Overall Mean (across 5 rates):")
for a in arms_order:
    mean_val = sum(rm_avg[r][a] for r in rates_order) / len(rates_order)
    print(f"  {a:<18}: {mean_val:5.2f}%")

# Overall Push-T Mean across rates
print("\nPush-T Overall Mean (across 5 rates):")
for a in arms_order:
    mean_val = sum(pusht_table[r][a] for r in rates_order) / len(rates_order)
    print(f"  {a:<18}: {mean_val:5.2f}%")

# Overall Grand Average across all 4 environments (Lift, Can, Square, Push-T) across all 5 rates
grand_avg_by_rate = {}
for r in rates_order:
    grand_avg_by_rate[r] = {}
    for a in arms_order:
        grand_avg_by_rate[r][a] = (rm_data['lift'][r][a][2] + rm_data['can'][r][a][2] + rm_data['square'][r][a][2] + pusht_table[r][a]) / 4.0

print("\n=== GRAND AVERAGE ACROSS BENCHMARKS (RoboMimic 3 Tasks + Push-T) ===")
for r in rates_order:
    print(f"{r:<8} | ARC: {grand_avg_by_rate[r]['arc']:5.1f}% | Spline: {grand_avg_by_rate[r]['spline']:5.1f}% | B-Spline: {grand_avg_by_rate[r]['bspline_eps_raw']:5.1f}% | TAC-Fold+Satfix: {grand_avg_by_rate[r]['tac_fold_satfix']:5.1f}% | QP-Anchor: {grand_avg_by_rate[r]['qp_anchor']:5.1f}%")

print("\n=== GRAND OVERALL MEAN (Across all rates and benchmarks) ===")
for a in arms_order:
    val = sum(grand_avg_by_rate[r][a] for r in rates_order) / len(rates_order)
    print(f"  {a:<18}: {val:5.2f}%")
