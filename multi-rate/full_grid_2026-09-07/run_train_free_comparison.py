"""Run the fixed evaluation-only campaign and rebuild its report after each cell."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_free_2026-09-28"
TASKS = ["PullCube-v1", "PickCube-v1", "PushCube-v1", "PokeCube-v1",
         "LiftPegUpright-v1", "StackCube-v1", "RollBall-v1", "AnymalC-Reach-v1"]
ARMS = ["native", "zoh", "tac_fold", "spline", "bspline_eps_raw",
        "tac_fold_satfix", "spline_satfix", "bspline_eps_satfix", "qp", "qp_anchor"]
SOURCES = ["eval_maniskill_batched.py", "harness.py", "dp_min.py", "resample_math.py",
           "resample_bspline2.py", "resample_qp.py", "PREREG_TRAIN_FREE_2026-09-28.md",
           "run_train_free_comparison.py"]


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mcnemar(a, b):
    x = sum(bool(u) and not v for u, v in zip(a, b))
    y = sum(bool(v) and not u for u, v in zip(a, b))
    n = x + y
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def holm(ps):
    adjusted = [0.] * len(ps)
    previous = 0.
    for rank, i in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        previous = max(previous, min(1., (len(ps) - rank) * ps[i]))
        adjusted[i] = previous
    return adjusted


def report():
    cells = []
    for task in TASKS:
        for k in (2, 4):
            path = OUT / f"{task}_k{k}.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            if d.get("status") != "complete":
                continue
            assert d["arms"] == ARMS and all(len(d["success"][a]) == d["n"] for a in ARMS)
            cells.append(d)
    lines = ["# Training-free resampler results", "",
             f"Completed: {len(cells)}/16 task/rate cells. Fresh paired batched closed-loop rollouts.",
             "Fixed seed-0 DP checkpoints; no resampler training. B-spline eps=.005.",
             "This saturation-screened ManiSkill suite does not establish a universal winner.", "",
             "| Task | k | n | " + " | ".join(ARMS) + " |",
             "|---|---|---|" + "---|" * len(ARMS)]
    for d in cells:
        lines.append(f"| {d['task']} | {d['k']} | {d['n']} | " + " | ".join(
            f"{100 * sum(d['success'][a]) / d['n']:.2f}%" for a in ARMS) + " |")
    contrasts = []
    if len(cells) == 16:
        lines += ["", "Equal-weight mean across all 16 cells (descriptive):", ""]
        for a in sorted(ARMS, key=lambda a: -sum(sum(d["success"][a]) / d["n"] for d in cells)):
            mean = 100 * sum(sum(d["success"][a]) / d["n"] for d in cells) / 16
            lines.append(f"- {a}: {mean:.2f}%")
        for family in [ARMS[2:5], ARMS[5:8]]:
            rows = []
            for d in cells:
                for a, b in itertools.combinations(family, 2):
                    rows.append(dict(task=d["task"], k=d["k"], a=a, b=b,
                                     delta_pp=100 * (sum(d["success"][a]) - sum(d["success"][b])) / d["n"],
                                     p=mcnemar(d["success"][a], d["success"][b])))
            for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
                row["holm_p"] = adjusted
            contrasts.extend(rows)
        (OUT / "contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
        lines += ["", "Exact paired McNemar with Holm correction: contrasts.json (separate 48-test raw/SATFix families)."]
    else:
        lines += ["", "Overall ranking withheld until all 16 cells are complete."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return len(cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert mcnemar([True] * 6, [False] * 6) == 0.03125
        assert mcnemar([True, False], [True, False]) == 1
        assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
        print("statistics self-check passed")
        return
    OUT.mkdir(exist_ok=True)
    if args.report_only:
        print(f"{report()}/16 complete")
        return
    manifest = {"sources": {f: sha(ROOT / f) for f in SOURCES}, "references": {}}
    for task in TASKS:
        for k in (2, 4):
            ref = ROOT / f"result_dp_{task}_k{k}_confirm.json"
            d = json.loads(ref.read_text())
            assert sha(ROOT / f"dp_{task}.pt") == d["checkpoint_sha256"]
            assert (ROOT / f"demos_{task}_200.npz").exists()
            manifest["references"][ref.name] = sha(ref)
    mf = OUT / "manifest.json"
    if mf.exists():
        assert json.loads(mf.read_text()) == manifest, "campaign inputs changed"
    else:
        mf.write_text(json.dumps(manifest, indent=2) + "\n")
    report()
    for task in TASKS:
        for k in (2, 4):
            out = OUT / f"{task}_k{k}.json"
            if out.exists():
                assert json.loads(out.read_text()).get("status") == "complete", f"incomplete file requires inspection: {out}"
                continue
            assert all(sha(ROOT / f) == digest for f, digest in manifest["sources"].items())
            command = [sys.executable, "-u", str(ROOT / "eval_maniskill_batched.py"), task,
                       "--reference", str(ROOT / f"result_dp_{task}_k{k}_confirm.json"),
                       "--output", str(out), "--workers", "8", "--arms", *ARMS]
            print(f"START {task} k={k}", flush=True)
            with (OUT / f"{task}_k{k}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, check=True)
            print(f"COMPLETE {report()}/16: {task} k={k}", flush=True)


if __name__ == "__main__":
    main()
