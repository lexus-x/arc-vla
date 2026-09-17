"""Aggregate multi_task_campaign.py's 21 (suite, task_id, arm) result files into the
per-task table + pooled two-proportion z-tests requested in the task brief. Extends
compare_smoke_arms.py's two_prop_z (copied verbatim, not reimplemented). Reports the
pre-registered grid (4 libero_spatial tasks + 1 task each from object/goal/libero_10)
and, separately and clearly labeled, pooling with the task_id=0 pilot
(smoke_results_v3.json / smoke_results_ppo.json), which was NOT pre-registered the
same way (it was the task-selected-for-non-degenerate-variance pilot).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
SUITE_TASK_IDS = {
    "libero_spatial": [1, 2, 4, 6],
    "libero_object": [3],
    "libero_goal": [3],
    "libero_10": [6],
}
ARMS = ["frozen", "grpo", "ppo"]


def two_prop_z(succ1, n1, succ2, n2):  # copied from compare_smoke_arms.py, unmodified
    p1, p2 = succ1 / n1, succ2 / n2
    p = (succ1 + succ2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else float("nan")
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return p2 - p1, z, p_value


def out_path(suite, task_id, arm):
    if suite == "libero_spatial":
        return ROOT / "logs" / f"campaign_task{task_id}_{arm}.json"
    return ROOT / "logs" / f"campaign_{suite}_task{task_id}_{arm}.json"


def load(suite, task_id, arm):
    p = out_path(suite, task_id, arm)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    print("=" * 100)
    print("PER-SUITE / PER-TASK RESULTS TABLE (pre-registered grid, n=20/arm)")
    print("=" * 100)
    counts = {}  # (suite, task_id, arm) -> (succ, n)
    all_records = {arm: [] for arm in ARMS}
    elapsed_by_key = {}
    missing = []

    for suite in SUITE_TASK_IDS:
        for task_id in SUITE_TASK_IDS[suite]:
            d0 = load(suite, task_id, "frozen")
            desc = d0["meta"]["task_description"] if d0 else "?"
            print(f"\n--- suite={suite} task_id={task_id}  ({desc!r}) ---")
            print(f"{'arm':<10}{'n':<6}{'succ':<6}{'rate':<8}{'mean_max_joint_speed':<22}"
                  f"{'mean_max_joint_jerk':<20}{'elapsed_s':<10}")
            for arm in ARMS:
                d = load(suite, task_id, arm)
                if d is None:
                    missing.append((suite, task_id, arm))
                    print(f"{arm:<10} MISSING")
                    continue
                recs = d["records"]
                succ = sum(1 for r in recs if r["success"])
                n = len(recs)
                counts[(suite, task_id, arm)] = (succ, n)
                all_records[arm].extend(recs)
                elapsed_by_key[(suite, task_id, arm)] = d["meta"]["elapsed_s"]
                mjs = sum(r["max_joint_speed"] for r in recs) / n
                mjj = sum(r["max_joint_jerk"] for r in recs) / n
                print(f"{arm:<10}{n:<6}{succ:<6}{succ/n:<8.3f}{mjs:<22.4f}{mjj:<20.4f}"
                      f"{d['meta']['elapsed_s']:<10.0f}")

    if missing:
        print(f"\nWARNING: missing result files: {missing}")

    total_elapsed = sum(elapsed_by_key.values())
    print(f"\nTotal summed job-seconds across all 21 runs: {total_elapsed:.0f}s "
          f"({total_elapsed/3600:.2f}h) -- NOT wall-clock (runs were parallelized across "
          f"workers); see the report's own wall-clock section for actual elapsed time.")

    print("\n" + "=" * 100)
    print("SAFETY METRIC (joint-velocity) SUMMARY, POOLED ACROSS ALL 21 RUNS, PER ARM")
    print("=" * 100)
    for arm in ARMS:
        recs = all_records[arm]
        if not recs:
            continue
        n = len(recs)
        mjs = [r["max_joint_speed"] for r in recs]
        mjj = [r["max_joint_jerk"] for r in recs]
        rho = [r["rho_safe"] for r in recs]
        viol = sum(1 for r in rho if r < 0) / n
        print(f"{arm:<8} n={n:<4} max_joint_speed: mean={sum(mjs)/n:.4f} max={max(mjs):.4f}  "
              f"max_joint_jerk: mean={sum(mjj)/n:.4f} max={max(mjj):.4f}  "
              f"violation_rate(rho<0, bound=2.0)={viol:.3f}")

    print("\n" + "=" * 100)
    print("POOLED TWO-PROPORTION Z-TESTS ACROSS ALL 6 PRE-REGISTERED TASKS "
          "(spatial 1/2/4/6 + object3 + goal3 + libero_10-6)")
    print("=" * 100)
    pooled = {}
    for arm in ARMS:
        succ_sum = sum(s for (su, t, a), (s, n) in counts.items() if a == arm)
        n_sum = sum(n for (su, t, a), (s, n) in counts.items() if a == arm)
        pooled[arm] = (succ_sum, n_sum)
        print(f"{arm:<10} succ={succ_sum:<6} n={n_sum:<6} rate={succ_sum/n_sum:.3f}" if n_sum else f"{arm}: no data")

    names = [a for a in ARMS if pooled.get(a, (0, 0))[1] > 0]
    print(f"\n{'comparison':<24}{'delta_pp':>10}{'z':>8}{'p':>9}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            (sa, na), (sb, nb) = pooled[a], pooled[b]
            delta, z, p = two_prop_z(sa, na, sb, nb)
            sig = "  <-- p<0.05" if p < 0.05 else ("  (encouraging, ns)" if p < 0.15 else "")
            print(f"{a+' vs '+b:<24}{delta*100:>+9.2f}{z:>8.2f}{p:>9.4f}{sig}")

    print("\n" + "=" * 100)
    print("PER-TASK (not pooled) z-tests, frozen vs grpo and frozen vs ppo -- does the")
    print("effect replicate on EACH individual pre-registered task?")
    print("=" * 100)
    print(f"{'suite/task':<26}{'comparison':<20}{'delta_pp':>10}{'z':>8}{'p':>9}")
    for suite in SUITE_TASK_IDS:
        for task_id in SUITE_TASK_IDS[suite]:
            key = f"{suite}/{task_id}"
            for arm in ["grpo", "ppo"]:
                if (suite, task_id, "frozen") not in counts or (suite, task_id, arm) not in counts:
                    continue
                sf, nf = counts[(suite, task_id, "frozen")]
                sa, na = counts[(suite, task_id, arm)]
                delta, z, p = two_prop_z(sf, nf, sa, na)
                sig = "  <-- p<0.05" if p < 0.05 else ("  (encouraging, ns)" if p < 0.15 else "")
                print(f"{key:<26}{'frozen vs '+arm:<20}{delta*100:>+9.2f}{z:>8.2f}{p:>9.4f}{sig}")

    # ---- Pooling with the task_id=0 pilot, clearly labeled separately ----
    pilot_path = ROOT / "smoke_results_v3.json"
    ppo_pilot_path = ROOT / "smoke_results_ppo.json"
    if pilot_path.exists():
        print("\n" + "=" * 100)
        print("POOLING WITH THE task_id=0 PILOT (smoke_results_v3.json / smoke_results_ppo.json,")
        print("libero_spatial only). PILOT WAS TASK-SELECTED POST-HOC (cherry-picked for")
        print("non-degenerate GRPO variance) -- reported separately, NOT folded silently into")
        print("the pre-registered pooled numbers above.")
        print("=" * 100)
        pv3 = json.loads(pilot_path.read_text())
        pilot_frozen = pv3["arm_A_frozen"]
        pilot_grpo = pv3["arm_B_naive_ttt"]
        pf_s, pf_n = sum(1 for r in pilot_frozen if r["success"]), len(pilot_frozen)
        pg_s, pg_n = sum(1 for r in pilot_grpo if r["success"]), len(pilot_grpo)
        print(f"pilot (task_id=0): frozen {pf_s}/{pf_n}={pf_s/pf_n:.3f}   grpo {pg_s}/{pg_n}={pg_s/pg_n:.3f}")
        pp_s, pp_n = 0, 0
        if ppo_pilot_path.exists():
            ppo_pilot = json.loads(ppo_pilot_path.read_text())["arm_C_ppo_ttt"]
            pp_s, pp_n = sum(1 for r in ppo_pilot if r["success"]), len(ppo_pilot)
            print(f"pilot (task_id=0): ppo {pp_s}/{pp_n}={pp_s/pp_n:.3f}")

        for arm, (ps, pn) in [("grpo", (pg_s, pg_n)), ("ppo", (pp_s, pp_n))]:
            if pn == 0 or pooled.get("frozen", (0, 0))[1] == 0 or pooled.get(arm, (0, 0))[1] == 0:
                continue
            comb_frozen = (pooled["frozen"][0] + pf_s, pooled["frozen"][1] + pf_n)
            comb_arm = (pooled[arm][0] + ps, pooled[arm][1] + pn)
            delta, z, p = two_prop_z(*comb_frozen, *comb_arm)
            print(f"\nfrozen+pilot vs {arm}+pilot (all 7 tasks: spatial 0,1,2,4,6 + object3 + "
                  f"goal3 + libero_10-6): frozen {comb_frozen[0]}/{comb_frozen[1]}="
                  f"{comb_frozen[0]/comb_frozen[1]:.3f}  {arm} {comb_arm[0]}/{comb_arm[1]}="
                  f"{comb_arm[0]/comb_arm[1]:.3f}  delta_pp={delta*100:+.2f}  z={z:.2f}  p={p:.4f}")
    else:
        print("\nNOTE: smoke_results_v3.json pilot file not found, skipping pilot-pooled section.")


if __name__ == "__main__":
    main()
