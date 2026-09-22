#!/usr/bin/env python3
"""Produce the preregistered aggregate and per-task result table."""

import argparse
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ("native", "zoh", "spline", "bspline", "tac_fold")


def mcnemar(left, right):
    left, right = list(map(bool, left)), list(map(bool, right))
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if not discordant:
        return left_only, right_only, 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(left_only, right_only) + 1))
    return left_only, right_only, min(1.0, 2 * tail / 2**discordant)


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * pvalues[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def pct(values):
    return 100 * sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "results.json")
    parser.add_argument("--output", type=Path, default=HERE / "REPORT.md")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    expected = data["episodes_per_task"]
    tasks = data["tasks"]
    if not args.allow_incomplete:
        assert len(tasks) == 20
        assert all(len(tasks[t]["success"][a]) == expected for t in tasks for a in ARMS)

    pooled = {arm: [] for arm in ARMS}
    rows = []
    for task, record in tasks.items():
        n = min(len(record["success"].get(arm, [])) for arm in ARMS)
        if not n:
            continue
        rates = {}
        for arm in ARMS:
            values = record["success"][arm][:n]
            pooled[arm].extend(values)
            rates[arm] = pct(values)
        rows.append((task, n, rates))

    tests = {}
    for reference in ("spline", "bspline"):
        tac_only, ref_only, pvalue = mcnemar(pooled["tac_fold"], pooled[reference])
        tests[reference] = {"tac_only": tac_only, "ref_only": ref_only, "p": pvalue}
    adjusted = holm({name: test["p"] for name, test in tests.items()})

    lines = [
        "# Newt × TAC-Fold: ManiSkill3-Panda20",
        "",
        "| Task | n | Native | ZOH | Cubic | B-spline | TAC-Fold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, n, rates in rows:
        lines.append("| " + " | ".join([
            task, str(n), *(f"{rates[arm]:.1f}%" for arm in ARMS)
        ]) + " |")
    lines.extend(["", "| Aggregate | Native | ZOH | Cubic | B-spline | TAC-Fold |", "|---|---:|---:|---:|---:|---:|"])
    macro = {arm: sum(row[2][arm] for row in rows) / len(rows) for arm in ARMS}
    lines.append("| Macro success | " + " | ".join(f"{macro[a]:.1f}%" for a in ARMS) + " |")
    lines.append("| Micro success | " + " | ".join(f"{pct(pooled[a]):.1f}%" for a in ARMS) + " |")
    lines.extend(["", "| Primary contrast | TAC-only | Reference-only | Delta | Raw p | Holm p |", "|---|---:|---:|---:|---:|---:|"])
    for reference, test in tests.items():
        delta = pct(pooled["tac_fold"]) - pct(pooled[reference])
        lines.append(f"| TAC-Fold vs {reference} | {test['tac_only']} | {test['ref_only']} | {delta:+.1f} pp | {test['p']:.4g} | {adjusted[reference]:.4g} |")
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
