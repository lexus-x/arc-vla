"""Audit ARC/TAC-Fold campaign JSONs; prints paired, Holm-corrected tables."""
import json
import hashlib
import math
from pathlib import Path

import numpy as np


EXPECTED = {
    "PushT-v1": (200, 400, 30_000),
    "lift": (200, 100, 30_000), "can": (200, 100, 30_000), "square": (200, 100, 30_000),
    "RC-TurnOffSinkFaucet": (39, 100, 15_000),
    "RC-CoffeePressButton": (39, 100, 15_000),
    "RC-TurnOffMicrowave": (39, 100, 15_000),
    "RC-CloseSingleDoor": (39, 100, 15_000),
}
RATES = (1, 2, 4)
HEADS = ("step", "arc")
STEP_ARMS = {"native", "zoh", "spline", "bspline_eps_raw", "tac_fold_satfix"}
ARC_ARMS = {"zoh", "spline", "bspline_eps_raw", "tac_fold", "tac_fold_satfix"}
VISUAL_ARMS = ARC_ARMS


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    ao, bo = int((a & ~b).sum()), int((~a & b).sum())
    n = ao + bo
    p = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(ao, bo) + 1)) / 2**n)
    return ao, bo, p


def holm(rows):
    ordered = sorted(enumerate(rows), key=lambda x: x[1]["p"])
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - rank) * row["p"]))
        rows[index]["p_holm"] = running


def load_results(root):
    found = {}
    checkpoint_by_head = {}
    for path in sorted(root.glob("result_*arcmain*.json")):
        data = json.loads(path.read_text())
        key = (data["task"], int(data["k"]), data["head"])
        if key in found:
            raise ValueError(f"duplicate protocol cell {key}: {found[key][0]} and {path}")
        if data.get("arc_training_rates") not in (None, [1, 2, 4]):
            raise ValueError(f"unexpected ARC training rates in {path}")
        expected = EXPECTED.get(data["task"])
        if expected is None or (data.get("n_train"), data.get("n"), data.get("train_steps")) != expected:
            raise ValueError(f"protocol budget mismatch in {path}")
        expected_arms = ARC_ARMS if data["head"] == "arc" else STEP_ARMS
        if set(data.get("arms", ())) != expected_arms:
            raise ValueError(f"arm mismatch in {path}")
        if set(data.get("success", ())) != expected_arms or any(
                len(data["success"][arm]) != expected[1] for arm in expected_arms):
            raise ValueError(f"incomplete success arrays in {path}")
        checkpoint = Path(data.get("checkpoint_path", ""))
        if not checkpoint.is_file() or sha256_file(checkpoint) != data.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint provenance mismatch in {path}")
        identity = (data["task"], data["head"])
        if identity in checkpoint_by_head and checkpoint_by_head[identity] != data["checkpoint_sha256"]:
            raise ValueError(f"rates did not reuse one checkpoint for {identity}")
        checkpoint_by_head[identity] = data["checkpoint_sha256"]
        found[key] = (path, data)
    expected_keys = {(task, rate, head) for task in EXPECTED for rate in RATES for head in HEADS}
    missing = sorted(expected_keys - set(found))
    if missing:
        raise ValueError(f"campaign coverage incomplete; missing {len(missing)} cells: {missing[:6]}")
    return found


def load_visual_results(root):
    visual_root = root / "stage1_arc_vision_results"
    checkpoint = visual_root / "CloseSingleDoor_arc.pt"
    if not checkpoint.is_file():
        raise ValueError(f"missing visual ARC checkpoint: {checkpoint}")
    checkpoint_hash = sha256_file(checkpoint)
    found = {}
    for rate in RATES:
        path = visual_root / f"CloseSingleDoor_arc_decimated_k{rate}.json"
        if not path.is_file():
            raise ValueError(f"missing visual ARC result: {path}")
        data = json.loads(path.read_text())
        if (data.get("task") != "CloseSingleDoor" or data.get("policy") != "visual_diffusion_policy" or
                data.get("head") != "arc" or data.get("k") != rate or data.get("complete") is not True or
                data.get("episode_count") != 15 or len(data.get("train_demo_indices", ())) != 35 or
                len(data.get("eval_demo_indices", ())) != 15 or data.get("eval_seed") != 0 or
                data.get("checkpoint_sha256") != checkpoint_hash):
            raise ValueError(f"visual protocol/provenance mismatch in {path}")
        budget = data.get("training_budget", {})
        if (budget.get("steps"), budget.get("batch_size"), budget.get("microbatch"),
                budget.get("seed"), budget.get("ddim_steps")) != (15_000, 256, 16, 0, 10):
            raise ValueError(f"visual training budget mismatch in {path}")
        if set(data.get("arms", ())) != VISUAL_ARMS or set(data.get("success", ())) != VISUAL_ARMS:
            raise ValueError(f"visual arm mismatch in {path}")
        if any(len(data["success"][arm]) != 15 for arm in VISUAL_ARMS):
            raise ValueError(f"incomplete visual success arrays in {path}")
        found[rate] = data
    return found


def paired_bootstrap_lower(a, b, draws=50_000):
    delta = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(0)
    sampled = []
    for size in (5_000,) * (draws // 5_000):
        sampled.append(delta[rng.integers(len(delta), size=(size, len(delta)))].mean(1))
    return 100 * float(np.quantile(np.concatenate(sampled), .05))


def compare_arrays(label, task, k, a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    ao, bo, p = mcnemar(a, b)
    ra, rb = np.mean(a), np.mean(b)
    return {"label": label, "task": task, "k": k, "n": len(a),
            "a": ra, "b": rb, "delta": 100 * (ra - rb), "a_only": ao, "b_only": bo,
            "p": p, "lower90": None,
            "void": (ra < .1 and rb < .1) or (ra > .9 and rb > .9)}


def compare(label, left, left_arm, right, right_arm):
    if left["task"] != right["task"] or left["k"] != right["k"] or left["n"] != right["n"]:
        raise ValueError(f"unpaired inputs for {label}")
    if left.get("eval_protocol") != right.get("eval_protocol") or left.get("eval_seed") != right.get("eval_seed"):
        raise ValueError(f"evaluation protocol mismatch for {label}")
    return compare_arrays(label, left["task"], left["k"],
                          left["success"][left_arm], right["success"][right_arm])


def print_table(title, rows):
    active = [row for row in rows if not row["void"]]
    holm(active)
    print(f"\n## {title}\n")
    print("| comparison | task | k | n | left | right | delta pp | 90% lower | discordant | p | Holm | verdict |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        verdict = "VOID" if row["void"] else ("left" if row["delta"] > 0 else "right" if row["delta"] < 0 else "tie")
        adjusted = "—" if row["void"] else f"{row['p_holm']:.4g}"
        lower = "—" if row["lower90"] is None else f"{row['lower90']:+.1f}"
        print(f"| {row['label']} | {row['task']} | {row['k']} | {row['n']} | {100*row['a']:.1f} | "
              f"{100*row['b']:.1f} | {row['delta']:+.1f} | {lower} | "
              f"{row['a_only']}/{row['b_only']} | "
              f"{row['p']:.4g} | {adjusted} | {verdict} |")


def print_visual(visual):
    rows = []
    for rate, data in visual.items():
        for baseline, label in (("zoh", "ZOH"), ("spline", "cubic spline"),
                                ("bspline_eps_raw", "B-spline")):
            rows.append(compare_arrays(
                f"visual ARC-TAC vs {label}", "CloseSingleDoor RGB+proprio", rate,
                data["success"]["tac_fold_satfix"], data["success"][baseline]))
    print_table("Visual-policy confirmation (supporting, n=15)", rows)


def main():
    root = Path(__file__).parent
    cells = load_results(root)
    visual = load_visual_results(root)
    primary_cells, mechanism, native = [], [], []
    tasks = sorted({task for task, _, _ in cells})
    for task in tasks:
        for k in (1, 2, 4):
            arc = cells.get((task, k, "arc"))
            step = cells.get((task, k, "step"))
            if not arc or not step:
                continue
            if k == 2:
                for baseline, label in (("tac_fold_satfix", "post-hoc TAC"),
                                        ("spline", "cubic spline"),
                                        ("bspline_eps_raw", "B-spline")):
                    row = compare(f"ARC-TAC vs {label}", arc[1], "tac_fold_satfix",
                                  step[1], baseline)
                    row["baseline"] = baseline
                    primary_cells.append(row)
            if k in (2, 4):
                mechanism.append(compare("ARC TAC vs ARC ZOH", arc[1], "tac_fold_satfix",
                                         arc[1], "zoh"))
            if k == 1:
                row = compare("ARC native-rate vs step native", arc[1], "tac_fold_satfix",
                              step[1], "native")
                row["lower90"] = paired_bootstrap_lower(
                    arc[1]["success"]["tac_fold_satfix"], step[1]["success"]["native"])
                native.append(row)
    primary = [row for row in primary_cells if row["task"] == "PushT-v1"]
    for baseline, label in (("tac_fold_satfix", "post-hoc TAC"),
                            ("spline", "cubic spline"),
                            ("bspline_eps_raw", "B-spline")):
        manipulation = [row for row in primary_cells
                        if row["task"] != "PushT-v1" and row["baseline"] == baseline and not row["void"]]
        if not manipulation:
            continue
        left, right = [], []
        for row in manipulation:
            arc = cells[(row["task"], 2, "arc")][1]
            step = cells[(row["task"], 2, "step")][1]
            left.extend(arc["success"]["tac_fold_satfix"])
            right.extend(step["success"][baseline])
        primary.append(compare_arrays(f"ARC-TAC vs {label}", "Manipulation pooled", 2, left, right))
    print_table("Primary: learned rate conditioning", primary)
    print_table("Supporting primary cells", primary_cells)
    print_table("Mechanism: TAC-Fold decoder", mechanism)
    print_table("Native-rate no-harm", native)
    pooled_arc, pooled_step = [], []
    for task in sorted(EXPECTED):
        pooled_arc.extend(cells[(task, 1, "arc")][1]["success"]["tac_fold_satfix"])
        pooled_step.extend(cells[(task, 1, "step")][1]["success"]["native"])
    pooled_native = compare_arrays(
        "ARC native-rate vs step native", "All state tasks pooled", 1, pooled_arc, pooled_step)
    pooled_native["lower90"] = paired_bootstrap_lower(pooled_arc, pooled_step)
    print_table("Native-rate pooled non-inferiority", [pooled_native])
    print_visual(visual)

    primary_pass = (len(primary) == 6 and all(
        not row["void"] and row["delta"] > 0 and row["p_holm"] < .05 for row in primary))
    mechanism_pass = any(
        not row["void"] and row["delta"] > 0 and row["p_holm"] < .05 for row in mechanism)
    native_pass = pooled_native["lower90"] > -5.0
    print("\n## Preregistered claim gates\n")
    print(f"- Primary six-comparison Holm gate: {'PASS' if primary_pass else 'FAIL'}")
    print(f"- TAC mechanism gate: {'PASS' if mechanism_pass else 'FAIL'}")
    print(f"- Native-rate pooled 90% lower bound > -5 pp: {'PASS' if native_pass else 'FAIL'} "
          f"({pooled_native['lower90']:+.1f} pp)")
    print(f"- Visual checkpoint/protocol gate: PASS ({sum(len(v['success']['tac_fold_satfix']) for v in visual.values())} paired rollouts)")
    print(f"- Overall configured positive claim: {'PASS' if primary_pass and mechanism_pass and native_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
