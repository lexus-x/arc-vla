"""VERIFY-stage stats for candidate G (progress-shaped GRPO). Compares the
4 post-training eval results (vanilla/pgr x seed0/seed1) on BINARY success,
pooled and per-seed, plus a paired McNemar test exploiting eval_pgr_checkpoints.py's
shared seed scheme (same env-reset seed used across all 4 checkpoints per
(task_id, episode index) -- matched initial conditions, not just matched n)."""
from __future__ import annotations

import json
from pathlib import Path

from pgad_stats import two_prop_z, mcnemar  # noqa: E402 -- reused, stdlib-only stats from this session's PGAD work

ROOT = Path(__file__).parent
RESULTS = ROOT / "pgr_eval_results"


def load(name):
    d = json.loads((RESULTS / f"{name}.json").read_text())
    return {(r["task_id"], r["episode"]): r["success"] for r in d["records"]}


def main():
    vanilla_s0, pgr_s0 = load("vanilla_s0"), load("pgr_s0")
    vanilla_s1, pgr_s1 = load("vanilla_s1"), load("pgr_s1")

    for tag, v, p in [("seed0", vanilla_s0, pgr_s0), ("seed1", vanilla_s1, pgr_s1)]:
        vk, vn = sum(v.values()), len(v)
        pk, pn = sum(p.values()), len(p)
        d, z, pval = two_prop_z(vk, vn, pk, pn)
        print(f"{tag}: vanilla {vk}/{vn}={vk/vn:.3f}  pgr {pk}/{pn}={pk/pn:.3f}  "
              f"delta={d*100:+.2f}pp  z={z:.3f}  p={pval:.4f}")
        n, b, c, stat, mp = mcnemar(v, p)
        print(f"  paired (matched seeds): n={n} vanilla_win={b} pgr_win={c} chi2={stat:.3f} p={mp:.4f}")

    print("\n--- pooled across both seeds ---")
    vk = sum(sum(v.values()) for v in [vanilla_s0, vanilla_s1])
    vn = len(vanilla_s0) + len(vanilla_s1)
    pk = sum(sum(p.values()) for p in [pgr_s0, pgr_s1])
    pn = len(pgr_s0) + len(pgr_s1)
    d, z, pval = two_prop_z(vk, vn, pk, pn)
    print(f"pooled: vanilla {vk}/{vn}={vk/vn:.3f}  pgr {pk}/{pn}={pk/pn:.3f}  "
          f"delta={d*100:+.2f}pp  z={z:.3f}  p={pval:.4f}")


if __name__ == "__main__":
    main()
