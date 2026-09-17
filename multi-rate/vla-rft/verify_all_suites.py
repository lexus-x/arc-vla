"""Two-proportion z-test, per-suite and pooled, over the 4 LIBERO suites'
1-seed baseline-vs-RL results. Stdlib only (normal approx, n=100/arm is large
enough). Numbers below are hand-copied from each suite's rl_vs_baseline.json /
PHASE3's same-seed row -- this script's CHECK re-derives them from those files
directly so a wrong copy-paste would fail the assert.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent


def two_prop_z(b_succ, b_n, r_succ, r_n):
    p1, p2 = b_succ / b_n, r_succ / r_n
    p = (b_succ + r_succ) / (b_n + r_n)
    se = math.sqrt(p * (1 - p) * (1 / b_n + 1 / r_n))
    z = (p2 - p1) / se if se > 0 else float("nan")
    p_value = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return p2 - p1, z, p_value


SUITES = {
    "spatial": dict(b_succ=72, b_n=100, r_succ=80, r_n=100),  # PHASE3 same-seed
    "object": dict(b_succ=86, b_n=100, r_succ=88, r_n=100),
    "goal": dict(b_succ=93, b_n=100, r_succ=91, r_n=100),
    "long": dict(b_succ=55, b_n=100, r_succ=58, r_n=100),
}

# CHECK: re-derive object/goal/long counts from disk, they must match the
# hardcoded values above (spatial's source predates this v2 pipeline, kept as
# documented in PHASE3).
for suite, path in {
    "object": "baseline_results/rl_object_seed0_v2/rl_vs_baseline.json",
    "goal": "baseline_results/rl_goal_seed0_v2/rl_vs_baseline.json",
    "long": "baseline_results/rl_long_seed0_v2/rl_vs_baseline.json",
}.items():
    d = json.loads((ROOT / path).read_text())
    got_b = round(d["baseline_aggregate"] * d["rl_n"])
    got_r = round(d["rl_aggregate"] * d["rl_n"])
    assert got_b == SUITES[suite]["b_succ"], (suite, "baseline", got_b)
    assert got_r == SUITES[suite]["r_succ"], (suite, "rl", got_r)

print(f"{'suite':<10} {'baseline':>10} {'RL':>10} {'delta_pp':>10} {'z':>7} {'p':>8}")
tot_b, tot_bn, tot_r, tot_rn = 0, 0, 0, 0
for suite, c in SUITES.items():
    delta, z, p = two_prop_z(c["b_succ"], c["b_n"], c["r_succ"], c["r_n"])
    print(f"{suite:<10} {c['b_succ']/c['b_n']*100:>9.1f}% {c['r_succ']/c['r_n']*100:>9.1f}% "
          f"{delta*100:>+9.2f} {z:>7.2f} {p:>8.4f}")
    tot_b += c["b_succ"]; tot_bn += c["b_n"]; tot_r += c["r_succ"]; tot_rn += c["r_n"]

delta, z, p = two_prop_z(tot_b, tot_bn, tot_r, tot_rn)
print("-" * 60)
print(f"{'pooled':<10} {tot_b/tot_bn*100:>9.1f}% {tot_r/tot_rn*100:>9.1f}% "
      f"{delta*100:>+9.2f} {z:>7.2f} {p:>8.4f}   (n={tot_bn}/arm)")
