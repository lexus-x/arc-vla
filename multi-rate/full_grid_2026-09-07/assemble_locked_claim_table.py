"""Build the locked paper table from explicit paired closed-loop result files."""
import hashlib
import json
import math
from pathlib import Path

SOURCES = {
    "PickCube-v1, 4X": "result_dp_PickCube-v1_k4_n400_anchor.json",
    "PushT-v1, 2X": "result_dp_PushT-v1_fair_n400.json",
}
ARMS = ("native", "zoh", "tac_fold_satfix", "qp", "qp_anchor")


def exact_mcnemar(a, b):
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    discordant = a_only + b_only
    if not discordant:
        return a_only, b_only, 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    return a_only, b_only, min(1.0, 2 * tail / 2**discordant)


def load_source(root, label, filename):
    path = root / filename
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data["n"] == 400, f"{filename}: expected n=400"
    assert data["head"] == "step", f"{filename}: expected step head"
    missing = set(ARMS) - data["success"].keys()
    assert not missing, f"{filename}: missing arms {sorted(missing)}"
    assert all(len(data["success"][arm]) == data["n"] for arm in ARMS)
    return data, {
        "label": label,
        "source": filename,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "task": data["task"],
        "policy": data["policy"],
        "rate_multiplier": data["k"],
        "n": data["n"],
        "checkpoint_sha256": data.get("checkpoint_sha256"),
        "eval_protocol": data.get("eval_protocol"),
        "eval_offset": data.get("eval_offset", 0),
        "n_train": data.get("n_train"),
        "train_steps": data.get("train_steps"),
    }


def main():
    root = Path(__file__).resolve().parent
    loaded = {label: load_source(root, label, filename) for label, filename in SOURCES.items()}
    manifest = {label: metadata for label, (_, metadata) in loaded.items()}
    (root / "paper_locked_results_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Locked-claim closed-loop results",
        "",
        "Every row is paired within one explicit n=400 result file. No globbing or largest-file selection is used.",
        "",
        "| Task / rate | native | ZOH | TAC-Fold+satfix | QP | QP-anchor | QP-anchor vs TAC-Fold | QP-anchor vs QP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, (data, _) in loaded.items():
        success = data["success"]
        rates = {arm: 100 * sum(success[arm]) / data["n"] for arm in ARMS}
        comparisons = []
        for ref in ("tac_fold_satfix", "qp"):
            a_only, ref_only, p = exact_mcnemar(success["qp_anchor"], success[ref])
            delta = rates["qp_anchor"] - rates[ref]
            comparisons.append(f"{delta:+.1f} pp, p={p:.3g} ({a_only}/{ref_only})")
        lines.append(
            f"| {label} | "
            + " | ".join(f"{rates[arm]:.1f}%" for arm in ARMS)
            + f" | {comparisons[0]} | {comparisons[1]} |"
        )
    lines += [
        "",
        "Discordant counts are shown as QP-anchor-only/reference-only. Exact two-sided McNemar p-values are uncorrected descriptive statistics; family-wise claims must use their preregistered Holm procedure.",
    ]
    (root / "PAPER_LOCKED_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
