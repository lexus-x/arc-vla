#!/usr/bin/env python3
"""Prepare and optionally run TypeSafe audits over this research campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORD = re.compile(r"[a-z0-9]+", re.I)
CLAIM_HINT = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%|\b(?:p\s*[<=>]|significant|best|beats?|loses?|ties?|"
    r"holds?|fails?|ceiling|floor|void|first|novel|outperform)\b)",
    re.I,
)
METHOD_ALIASES = {
    "tac-fold": "tac_fold_satfix",
    "tac_fold": "tac_fold_satfix",
    "cubic": "cubic_spline_satfix",
    "spline": "spline_satfix",
    "b-spline": "bspline_eps_satfix",
    "zoh": "zoh",
    "qp": "qp",
    "native": "native",
}


def tokens(text: str) -> set[str]:
    return {word.lower() for word in WORD.findall(text) if len(word) > 2}


def exact_mcnemar(left: list[Any], right: list[Any]) -> dict[str, Any]:
    """Two-sided exact binomial McNemar test, matching the campaign scripts."""
    left_only = sum(bool(a) and not bool(b) for a, b in zip(left, right))
    right_only = sum(not bool(a) and bool(b) for a, b in zip(left, right))
    discordant = left_only + right_only
    p = 1.0 if not discordant else min(
        1.0,
        2
        * sum(math.comb(discordant, i) for i in range(min(left_only, right_only) + 1))
        / 2**discordant,
    )
    return {"left_only": left_only, "right_only": right_only, "p_raw": p}


def load_result(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    success = data.get("success")
    if not isinstance(success, dict) or not success:
        return None
    arms = {name: values for name, values in success.items() if isinstance(values, list)}
    lengths = {len(values) for values in arms.values()}
    if len(arms) < 2 or len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        return None
    n = next(iter(lengths))
    return {
        "source": str(path.relative_to(ROOT)),
        "task": data.get("task", path.stem),
        "policy": data.get("policy", "replay" if "replay" in path.name else "unknown"),
        "k": data.get("k"),
        "n": n,
        "success": arms,
    }


def result_catalog(root: Path) -> list[dict[str, Any]]:
    results = []
    patterns = ("result_*.json", "*replay_*.json")
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            item = load_result(path)
            if not item:
                continue
            arms = item.pop("success")
            rates = {name: 100 * sum(map(bool, values)) / item["n"] for name, values in arms.items()}
            comparisons = []
            names = list(arms)
            for i, left_name in enumerate(names):
                for right_name in names[i + 1 :]:
                    test = exact_mcnemar(arms[left_name], arms[right_name])
                    comparisons.append(
                        {
                            "left": left_name,
                            "right": right_name,
                            "delta_pp": rates[left_name] - rates[right_name],
                            **test,
                        }
                    )
            results.append({**item, "success_pct": rates, "comparisons": comparisons})
    return results


def markdown_claims(path: Path) -> list[dict[str, Any]]:
    claims = []
    source = str(path.relative_to(ROOT))
    buffered: list[str] = []
    start = 1
    section = "preamble"

    def add(lines: list[str], line_number: int) -> None:
        text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", " ".join(lines)).strip(" |\t")
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text):
            if len(sentence) >= 20 and CLAIM_HINT.search(sentence):
                claims.append(
                    {"source": source, "line": line_number, "section": section, "context": text, "text": sentence}
                )

    for number, line in enumerate(path.read_text().splitlines() + [""], 1):
        stripped = line.strip()
        boundary = not stripped or stripped.startswith("#") or stripped.startswith("|")
        new_item = stripped.startswith(("- ", "* "))
        if boundary or new_item:
            if buffered:
                add(buffered, start)
                buffered = []
            if stripped.startswith("#"):
                section = stripped.lstrip("# ")
            if stripped.startswith("|"):
                add([stripped], number)
            elif new_item:
                buffered, start = [stripped], number
            continue
        if not buffered:
            start = number
        buffered.append(stripped)
    return claims


def relevance(text: str, item: dict[str, Any]) -> int:
    haystack = " ".join(
        [item["source"], str(item.get("task", "")), str(item.get("policy", "")), str(item.get("k", "")), *item["success_pct"]]
    ).lower()
    score = len(tokens(text) & tokens(haystack))
    lowered = text.lower()
    task_name = str(item.get("task", "")).lower().removesuffix("-v1")
    if task_name and task_name in lowered.replace(" ", ""):
        score += 10
    if item.get("k") is not None and re.search(rf"\bk\s*=\s*{item['k']}\b", lowered):
        score += 4
    if "open-loop" in lowered and "replay" in item["source"]:
        score += 5
    if "closed-loop" in lowered and item["source"].startswith("result_"):
        score += 5
    if "flow matching" in lowered and item["source"].startswith("result_fm_"):
        score += 8
    for alias, arm in METHOD_ALIASES.items():
        if alias in lowered and arm in item["success_pct"]:
            score += 2
    return score


def claim_relevance(claim: dict[str, Any], item: dict[str, Any]) -> int:
    """Favor the exact sentence while retaining paragraph context for pronouns and policy names."""
    return 3 * relevance(f"{claim['section']} {claim['text']}", item) + relevance(claim["context"], item)


def canonical_source(item: dict[str, Any]) -> int:
    name = item["source"]
    if name.startswith(("qp_replay_", "qp2_replay_")):
        return 2
    if name.startswith(("result_dp_", "result_fm_")) and not any(
        marker in name for marker in ("anchor", "noise", "smoke", "matrix", "eval", "blocksum")
    ):
        return 1
    return 0


def compact_evidence(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item[key] for key in ("source", "task", "policy", "k", "n", "success_pct", "comparisons")
    }


def prereg_sections(root: Path) -> list[dict[str, Any]]:
    sections = []
    for path in sorted(root.glob("PREREG*.md")):
        heading = "preamble"
        lines: list[str] = []
        start = 1
        for number, line in enumerate(path.read_text().splitlines() + ["# END"], 1):
            if line.startswith("#"):
                if lines:
                    sections.append(
                        {
                            "source": str(path.relative_to(ROOT)),
                            "heading": heading,
                            "line": start,
                            "text": "\n".join(lines).strip(),
                        }
                    )
                heading, lines, start = line.lstrip("# "), [], number + 1
            else:
                lines.append(line)
    return [section for section in sections if section["text"]]


def closest(text: str, items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    wanted = tokens(text)
    def item_tokens(item: dict[str, Any]) -> set[str]:
        return tokens(" ".join(map(str, item.values())))

    ranked = sorted(items, key=lambda item: len(wanted & item_tokens(item)), reverse=True)
    return [item for item in ranked[:limit] if wanted & item_tokens(item)]


def literature_entries(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    matches = list(re.finditer(r"(?m)^###\s+(\d+)\.\s+(.+)$", text))
    entries = []
    for match, nxt in zip(matches, matches[1:] + [None]):
        body = text[match.end() : nxt.start() if nxt else len(text)].strip()
        entries.append(
            {
                "source": str(path.relative_to(ROOT)),
                "number": int(match.group(1)),
                "title": match.group(2).strip(),
                "text": body,
            }
        )
    return entries


def deterministic_triage(item: dict[str, Any]) -> dict[str, Any]:
    excluded = {"native", "original"}
    rates = {name: value for name, value in item["success_pct"].items() if name not in excluded}
    low, high = min(rates.values()), max(rates.values())
    if high < 10:
        tag = "floor_candidate"
    elif low > 90:
        tag = "ceiling_candidate"
    else:
        tag = "informative_candidate"
    return {**{key: item[key] for key in ("source", "task", "policy", "k", "n")}, "rates": rates, "tag": tag}


def question_specs() -> dict[str, Any]:
    return {
        "claim_support": {
            "type": "choice",
            "instructions": "How does the supplied evidence relate to the research claim? Check method identity, task, policy, k, n, direction, magnitude, and whether p is raw or adjusted.",
            "criteria": {
                "supported": "The evidence supports every material part of the claim.",
                "contradicted": "The evidence directly disagrees with at least one material part.",
                "unsupported": "The evidence does not establish the claim.",
                "wrong_attribution": "The numbers exist but belong to another method, comparison, task, regime, or statistical family.",
            },
        },
        "claim_strength": {
            "type": "choice",
            "instructions": "Is the wording calibrated to the supplied evidence and its scope?",
            "criteria": {
                "calibrated": "Strength and scope match the evidence.",
                "overstated": "Wording generalizes beyond tasks, regimes, correction status, or evidence strength.",
                "understated": "Wording omits a clearly supported material result.",
                "unclear": "The wording or supplied evidence is too ambiguous to judge.",
            },
        },
        "preregistered_endpoint": {
            "type": "noul",
            "instructions": "Do the preregistration excerpts explicitly cover the claim's endpoint, task, method comparison, and regime?",
        },
        "preregistered_sample": {
            "type": "noul",
            "instructions": "Do the preregistration excerpts support the sample size or stopping condition asserted by the claim?",
        },
        "preregistered_correction": {
            "type": "noul",
            "instructions": "Do the preregistration excerpts support the multiple-testing family and correction status asserted by the claim?",
        },
        "confirmatory_language": {
            "type": "noul",
            "instructions": "Is it justified to present this claim as confirmatory rather than exploratory, given the preregistration excerpts?",
        },
        "triage_interpretation": {
            "type": "choice",
            "instructions": "Which bounded interpretation best describes this result cell? Do not infer a mechanism not present in the state.",
            "criteria": {
                "floor_or_collapse": "Decimated arms are too unsuccessful for method ranking to be informative.",
                "ceiling_or_parity": "Success is too high or differences too small for useful method separation.",
                "method_separation": "The cell contains potentially informative method differences.",
                "insufficient_context": "The numeric summary alone cannot support one of the other labels.",
            },
        },
        "literature_relevance": {
            "type": "score",
            "instructions": "How directly does this source address multi-rate robot-policy action representations or reconstruction?",
            "criteria": ["unrelated", "background only", "adjacent method", "direct comparator or foundation"],
        },
        "literature_overlap": {
            "type": "score",
            "instructions": "How much does the described method overlap ARC--TAC's resolution conditioning, block-integral targets, or conservative bounded reconstruction?",
            "criteria": ["no overlap", "one broad theme", "one concrete component", "multiple defining components"],
        },
        "literature_quality": {
            "type": "score",
            "instructions": "How strong is this entry as evidence for a paper claim, using only the supplied provenance and method description?",
            "criteria": ["unusable", "weak or incomplete provenance", "credible supporting source", "strong primary source"],
        },
        "literature_primary": {
            "type": "noul",
            "instructions": "Does the entry describe and link an original primary research artifact rather than a secondary summary?",
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    args.document = args.document.resolve()
    args.literature = args.literature.resolve()
    catalog = result_catalog(ROOT)
    sections = prereg_sections(ROOT)
    claims = markdown_claims(args.document)
    claim_jobs = []
    for claim in claims[: args.max_claims]:
        query = " ".join((claim["section"], claim["context"], claim["text"]))
        ranked = sorted(
            catalog,
            key=lambda item: claim_relevance(claim, item) + 5 * canonical_source(item),
            reverse=True,
        )
        evidence = [compact_evidence(item) for item in ranked[: args.evidence_limit] if claim_relevance(claim, item)]
        claim_jobs.append(
            {
                "id": f"claim:{claim['source']}:{claim['line']}",
                "workflow": "claim_and_prereg_audit",
                "state": {
                    "claim": claim,
                    "candidate_evidence": evidence,
                    "candidate_preregistration": closest(query, sections, args.prereg_limit),
                    "policy": {
                        "raw_p_is_not_holm_adjusted": True,
                        "open_and_closed_loop_are_not_interchangeable": True,
                        "confirmatory_claims_require_preregistered_threshold_and_n": True,
                    },
                },
                "questions": [
                    "claim_support",
                    "claim_strength",
                    "preregistered_endpoint",
                    "preregistered_sample",
                    "preregistered_correction",
                    "confirmatory_language",
                ],
            }
        )
    triage_catalog = sorted(
        catalog,
        key=lambda item: max((claim_relevance(claim, item) for claim in claims), default=0)
        + 5 * canonical_source(item),
        reverse=True,
    )
    triage_jobs = [
        {
            "id": f"triage:{item['source']}",
            "workflow": "experiment_triage",
            "state": deterministic_triage(item),
            "questions": ["triage_interpretation"],
        }
        for item in triage_catalog[: args.max_triage]
    ]
    literature_jobs = [
        {
            "id": f"literature:{entry['number']}",
            "workflow": "related_work_screen",
            "state": entry,
            "questions": ["literature_relevance", "literature_overlap", "literature_quality", "literature_primary"],
        }
        for entry in literature_entries(args.literature)[: args.max_literature]
    ]
    return {
        "schema_version": 1,
        "generated_from": {"document": str(args.document), "literature": str(args.literature)},
        "question_specs": question_specs(),
        "counts": {
            "results": len(catalog),
            "claims": len(claim_jobs),
            "triage": len(triage_jobs),
            "literature": len(literature_jobs),
        },
        "jobs": claim_jobs + triage_jobs + literature_jobs,
    }


def build_question(spec: dict[str, Any]) -> Any:
    from typesafe_sdk import Choice, Noul, Score

    constructors = {"choice": Choice, "noul": Noul, "score": Score}
    return constructors[spec["type"]](
        **{key: value for key, value in spec.items() if key != "type"}
    )


def run_bundle(bundle: dict[str, Any], confidence: float) -> dict[str, Any]:
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError as error:
        raise SystemExit("Install the live client first: pip install typesafe-sdk") from error
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is required for live TypeSafe evaluation")
    specs = bundle["question_specs"]
    results = []
    with TypeSafeClient() as client:
        for job in bundle["jobs"]:
            questions = {name: build_question(specs[name]) for name in job["questions"]}
            response = client.system_one(state=job["state"], questions=questions)
            payload = response.model_dump(mode="json")
            review = []
            for name, answer in payload.get("answers", {}).items():
                certainty = answer.get("confidence")
                if certainty is None and "noul" in answer:
                    probability = float(answer["noul"])
                    certainty = abs(probability - 0.5) * 2
                if certainty is not None and certainty < confidence:
                    review.append(name)
            results.append({"id": job["id"], "workflow": job["workflow"], "review": review, "response": payload})
    return {"schema_version": 1, "confidence_threshold": confidence, "results": results}


def self_test() -> None:
    assert exact_mcnemar([1, 1, 0, 0], [1, 0, 1, 0]) == {
        "left_only": 1,
        "right_only": 1,
        "p_raw": 1.0,
    }
    assert deterministic_triage(
        {"source": "x", "task": "t", "policy": "p", "k": 4, "n": 2, "success_pct": {"native": 50, "zoh": 2, "tac": 3}}
    )["tag"] == "floor_candidate"
    sample = ROOT / "result_dp_PushT-v1.json"
    if sample.exists():
        loaded = load_result(sample)
        assert loaded and loaded["n"] == len(loaded["success"]["zoh"])
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="create an inspectable offline audit bundle")
    prep.add_argument("--document", type=Path, default=ROOT / "RESULTS_QP_DRAFT.md")
    prep.add_argument(
        "--literature",
        type=Path,
        default=ROOT / "paper_arc_tac/phase1_literature/LITERATURE_SEARCH_REPORT.md",
    )
    prep.add_argument("--output", type=Path, default=ROOT / "typesafe_audit_bundle.json")
    prep.add_argument("--max-claims", type=int, default=60)
    prep.add_argument("--max-literature", type=int, default=40)
    prep.add_argument("--max-triage", type=int, default=40)
    prep.add_argument("--evidence-limit", type=int, default=3)
    prep.add_argument("--prereg-limit", type=int, default=3)
    live = sub.add_parser("run", help="run a prepared bundle through TypeSafe")
    live.add_argument("bundle", type=Path)
    live.add_argument("--output", type=Path, default=ROOT / "typesafe_audit_report.json")
    live.add_argument("--confidence", type=float, default=0.8)
    sub.add_parser("self-test", help="run the smallest local regression check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-test":
        self_test()
        return
    if args.command == "prepare":
        bundle = prepare(args)
        args.output.write_text(json.dumps(bundle, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), **bundle["counts"]}, indent=2))
        return
    bundle = json.loads(args.bundle.read_text())
    report = run_bundle(bundle, args.confidence)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "results": len(report["results"])}, indent=2))


if __name__ == "__main__":
    main()
