"""VERIFY-stage stats for the PGAD toolkit paper. Compares the two new arms
(gated, always_careful) against the EXISTING frozen baseline (not rerun --
see FRAME.md), using the exact seed-pairing built into run_pgad_eval.py.

Reports, in order of importance to the paper's MATTERS claim:
  1. Ceiling-task check (baseline >=90%): does gated regress? (the ATTC/RTCF question)
  2. Weak-task check (baseline <90%): does gated help?
  3. Pooled two-proportion z-test (gated vs frozen), matching this project's
     existing stats convention (PHASE4/5/6).
  4. Paired McNemar's test (gated vs frozen), which is the statistically
     correct test here since every episode is seed-matched 1:1 -- more
     powerful than the unpaired z-test when it applies.
  5. always_careful vs frozen, to isolate whether GATING (not just "being more
     careful") is what's doing any work (CAUSED-stage ablation).

Stdlib only, matching vla-research skill's scripts/compare_results.py convention.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
VLA_RFT_LOGS = Path("/home/user/Desktop/multi-rate/vla-rft/logs")
PGAD_RESULTS = ROOT / "pgad_results"

SUITE_TASK_IDS = {
    "libero_spatial": [7, 8],
    "libero_object": [0, 1],
    "libero_goal": [0, 1],
    "libero_10": [0, 1],
}


def frozen_path(suite, task_id):
    if suite == "libero_spatial":
        return VLA_RFT_LOGS / f"campaign_task{task_id}_frozen.json"
    return VLA_RFT_LOGS / f"campaign_{suite}_task{task_id}_frozen.json"


def pgad_path(suite, task_id, arm):
    return PGAD_RESULTS / f"pgad_{suite}_task{task_id}_{arm}.json"


def load_successes(path):
    """Returns {seed: success_bool} preserving pairing."""
    d = json.loads(Path(path).read_text())
    return {r["seed"]: bool(r["success"]) for r in d["records"]}


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p2 - p1, float("nan"), 1.0
    z = (p2 - p1) / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p2 - p1, z, p_value


def mcnemar(base: dict, var: dict):
    """Paired test on shared seeds. b = base success, var fail; c = base fail, var success."""
    shared = sorted(set(base) & set(var))
    b = sum(1 for s in shared if base[s] and not var[s])
    c = sum(1 for s in shared if not base[s] and var[s])
    n = b + c
    if n == 0:
        return len(shared), b, c, float("nan"), 1.0
    stat = (abs(b - c) - 1) ** 2 / n  # continuity-corrected chi-square, 1 df
    # chi-square(1) survival function via erf (chi2_1 = z^2 for standard normal z)
    z = math.sqrt(stat)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return len(shared), b, c, stat, p_value


def main():
    rows = []
    for suite, task_ids in SUITE_TASK_IDS.items():
        for task_id in task_ids:
            fp = frozen_path(suite, task_id)
            if not fp.exists():
                print(f"MISSING frozen baseline: {fp}")
                continue
            frozen = load_successes(fp)
            row = {"suite": suite, "task_id": task_id,
                   "frozen_n": len(frozen), "frozen_k": sum(frozen.values())}
            for arm in ("gated", "always_careful"):
                gp = pgad_path(suite, task_id, arm)
                if not gp.exists():
                    row[arm] = None
                    continue
                var = load_successes(gp)
                row[arm] = var
            rows.append(row)

    print("=" * 100)
    print(f"{'suite':16} {'task':5} {'frozen':>10} {'gated':>10} {'always_careful':>16}   baseline class")
    pooled_frozen_k = pooled_frozen_n = pooled_gated_k = pooled_gated_n = 0
    pooled_ac_k = pooled_ac_n = 0
    ceiling_rows, weak_rows = [], []
    all_frozen_gated_pairs = {}
    for row in rows:
        fn, fk = row["frozen_n"], row["frozen_k"]
        frate = fk / fn
        gated = row.get("gated")
        ac = row.get("always_careful")
        gk = sum(gated.values()) if isinstance(gated, dict) else None
        gn = len(gated) if isinstance(gated, dict) else None
        ack = sum(ac.values()) if isinstance(ac, dict) else None
        acn = len(ac) if isinstance(ac, dict) else None
        cls = "CEILING (>=90%)" if frate >= 0.90 else "weak (<90%)"
        gstr = f"{gk}/{gn}" if gn else "n/a"
        astr = f"{ack}/{acn}" if acn else "n/a"
        print(f"{row['suite']:16} {row['task_id']:<5} {fk}/{fn:<7} {gstr:>10} {astr:>16}   {cls}")
        (ceiling_rows if frate >= 0.90 else weak_rows).append(row)
        pooled_frozen_k += fk; pooled_frozen_n += fn
        if gn:
            pooled_gated_k += gk; pooled_gated_n += gn
            key = f"{row['suite']}_t{row['task_id']}"
            fp_seeds = load_successes(frozen_path(row["suite"], row["task_id"]))
            all_frozen_gated_pairs.update({(key, s): (fp_seeds[s], gated[s]) for s in gated if s in fp_seeds})
        if acn:
            pooled_ac_k += ack; pooled_ac_n += acn
    print("=" * 100)

    def class_summary(name, class_rows):
        fk = sum(r["frozen_k"] for r in class_rows); fn = sum(r["frozen_n"] for r in class_rows)
        gk = sum(sum(r["gated"].values()) for r in class_rows if r.get("gated"))
        gn = sum(len(r["gated"]) for r in class_rows if r.get("gated"))
        if fn == 0 or gn == 0:
            print(f"{name}: no data")
            return
        d, z, p = two_prop_z(fk, fn, gk, gn)
        print(f"{name}: frozen {fk}/{fn}={fk/fn:.3f}  gated {gk}/{gn}={gk/gn:.3f}  "
              f"delta={d*100:+.2f}pp  z={z:.3f} p={p:.4f}")

    print("\n--- 1. CEILING-task check (does gating regress?) ---")
    class_summary("ceiling", ceiling_rows)
    print("\n--- 2. weak-task check (does gating help?) ---")
    class_summary("weak", weak_rows)

    print("\n--- 3. Pooled two-proportion z-test, gated vs frozen ---")
    if pooled_gated_n:
        d, z, p = two_prop_z(pooled_frozen_k, pooled_frozen_n, pooled_gated_k, pooled_gated_n)
        print(f"frozen {pooled_frozen_k}/{pooled_frozen_n}={pooled_frozen_k/pooled_frozen_n:.4f}  "
              f"gated {pooled_gated_k}/{pooled_gated_n}={pooled_gated_k/pooled_gated_n:.4f}  "
              f"delta={d*100:+.2f}pp  z={z:.3f}  p={p:.4f}")

    print("\n--- 4. Paired McNemar's test, gated vs frozen (pooled across all seed-shared episodes) ---")
    base_all, var_all = {}, {}
    for (key, s), (fsucc, gsucc) in all_frozen_gated_pairs.items():
        base_all[(key, s)] = fsucc
        var_all[(key, s)] = gsucc
    n, b, c, stat, p = mcnemar(base_all, var_all)
    print(f"n_pairs={n}  frozen_win_gated_lose(b)={b}  gated_win_frozen_lose(c)={c}  "
          f"chi2={stat:.3f}  p={p:.4f}")

    print("\n--- 5. Pooled two-proportion z-test, always_careful vs frozen (isolates gating vs. blanket carefulness) ---")
    if pooled_ac_n:
        d, z, p = two_prop_z(pooled_frozen_k, pooled_frozen_n, pooled_ac_k, pooled_ac_n)
        print(f"frozen {pooled_frozen_k}/{pooled_frozen_n}={pooled_frozen_k/pooled_frozen_n:.4f}  "
              f"always_careful {pooled_ac_k}/{pooled_ac_n}={pooled_ac_k/pooled_ac_n:.4f}  "
              f"delta={d*100:+.2f}pp  z={z:.3f}  p={p:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
