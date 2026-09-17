"""Two-proportion z-test comparison across smoke-test arms (frozen / GRPO / PPO-style),
same style as verify_all_suites.py. Re-derives success counts directly from each arm's
JSON file -- no hardcoded numbers, so a stale copy-paste can't silently go unnoticed.

Arms compared:
  frozen   <- smoke_results_v3.json["arm_A_frozen"]
  grpo     <- smoke_results_v3.json["arm_B_naive_ttt"]
  ppo      <- smoke_ttvla_ppo_results.json["arm_C_ppo"]  (once it exists)
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent


def two_prop_z(succ1, n1, succ2, n2):
    p1, p2 = succ1 / n1, succ2 / n2
    p = (succ1 + succ2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else float("nan")
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return p2 - p1, z, p_value


def load_arm_counts(path, key):
    d = json.loads((ROOT / path).read_text())
    recs = d[key]
    succ = sum(1 for r in recs if r["success"])
    return succ, len(recs)


def main():
    v3 = "smoke_results_v3.json"
    arms = {}
    arms["frozen"] = load_arm_counts(v3, "arm_A_frozen")
    arms["grpo"] = load_arm_counts(v3, "arm_B_naive_ttt")

    ppo_path = ROOT / "smoke_results_ppo.json"
    if ppo_path.exists():
        d = json.loads(ppo_path.read_text())
        # accept a couple of plausible key names since the building agent may vary this
        for cand_key in ("arm_C_ppo_ttt", "arm_C_ppo", "arm_ppo", "ppo"):
            if cand_key in d:
                recs = d[cand_key]
                arms["ppo"] = (sum(1 for r in recs if r["success"]), len(recs))
                break
        else:
            print(f"WARNING: {ppo_path} exists but no recognized ppo key found "
                  f"(looked for arm_C_ppo/arm_ppo/ppo); top-level keys: {list(d.keys())}")
    else:
        print(f"NOTE: {ppo_path} not found yet -- comparing only frozen vs grpo for now. "
              f"Re-run this script once the PPO arm's results file lands.")

    print(f"\n{'arm':<10}{'succ':<8}{'n':<6}{'rate':<8}")
    for name, (s, n) in arms.items():
        print(f"{name:<10}{s:<8}{n:<6}{s/n:<8.3f}")

    names = list(arms.keys())
    print(f"\n{'comparison':<20}{'delta_pp':>10}{'z':>8}{'p':>9}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            (sa, na), (sb, nb) = arms[a], arms[b]
            delta, z, p = two_prop_z(sa, na, sb, nb)
            sig = "  <-- p<0.05" if p < 0.05 else ("  (encouraging, ns)" if p < 0.15 else "")
            print(f"{a+' vs '+b:<20}{delta*100:>+9.2f}{z:>8.2f}{p:>9.4f}{sig}")


if __name__ == "__main__":
    main()
