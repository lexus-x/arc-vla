import hashlib
import json
from pathlib import Path

import pytest

import analyze_arc_tacfold as analysis


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_synthetic_campaign(root: Path):
    for task, (n_train, n_eval, steps) in analysis.EXPECTED.items():
        protocol = "random_reset" if task.startswith("RC-") else "heldout_demo_state"
        for head in analysis.HEADS:
            checkpoint = root / f"dp_{task}_{head}_arcmain.pt"
            checkpoint.write_bytes(f"{task}:{head}".encode())
            arms = analysis.ARC_ARMS if head == "arc" else analysis.STEP_ARMS
            for rate in analysis.RATES:
                success = {}
                for arm in arms:
                    if head == "arc":
                        success[arm] = [arm == "tac_fold_satfix"] * n_eval
                    else:
                        success[arm] = ([True] * n_eval if rate == 1 and arm == "native"
                                        else [False] * n_eval)
                payload = {
                    "task": task, "policy": "dp", "k": rate, "head": head,
                    "n": n_eval, "n_train": n_train, "train_steps": steps,
                    "arms": sorted(arms), "success": success,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": digest(checkpoint),
                    "eval_protocol": protocol, "eval_seed": 0,
                }
                if head == "arc":
                    payload["arc_training_rates"] = [1, 2, 4]
                name = f"result_{task}_{head}_k{rate}_arcmain.json"
                (root / name).write_text(json.dumps(payload))

    visual_root = root / "stage1_arc_vision_results"
    visual_root.mkdir()
    checkpoint = visual_root / "CloseSingleDoor_arc.pt"
    checkpoint.write_bytes(b"visual checkpoint")
    for rate in analysis.RATES:
        success = {arm: [arm == "tac_fold_satfix"] * 15 for arm in analysis.VISUAL_ARMS}
        payload = {
            "stage": 1, "task": "CloseSingleDoor", "policy": "visual_diffusion_policy",
            "head": "arc", "k": rate, "complete": True, "episode_count": 15,
            "train_demo_indices": list(range(35)), "eval_demo_indices": list(range(35, 50)),
            "eval_seed": 0, "checkpoint_sha256": digest(checkpoint),
            "training_budget": {"steps": 15_000, "batch_size": 256, "microbatch": 16,
                                "seed": 0, "ddim_steps": 10},
            "arms": sorted(analysis.VISUAL_ARMS), "success": success,
        }
        (visual_root / f"CloseSingleDoor_arc_decimated_k{rate}.json").write_text(
            json.dumps(payload))


def test_analyzer_requires_complete_campaign_and_visual_provenance(tmp_path: Path):
    with pytest.raises(ValueError, match="coverage incomplete"):
        analysis.load_results(tmp_path)

    write_synthetic_campaign(tmp_path)
    cells = analysis.load_results(tmp_path)
    visual = analysis.load_visual_results(tmp_path)

    assert len(cells) == len(analysis.EXPECTED) * len(analysis.RATES) * len(analysis.HEADS)
    assert set(visual) == set(analysis.RATES)


def test_analyzer_reports_all_positive_claim_gates_from_validated_inputs(
        tmp_path: Path, monkeypatch, capsys):
    write_synthetic_campaign(tmp_path)
    monkeypatch.setattr(analysis, "__file__", str(tmp_path / "analyze_arc_tacfold.py"))

    analysis.main()

    output = capsys.readouterr().out
    assert "Primary six-comparison Holm gate: PASS" in output
    assert "TAC mechanism gate: PASS" in output
    assert "Native-rate pooled 90% lower bound > -5 pp: PASS" in output
    assert "Visual checkpoint/protocol gate: PASS" in output
    assert "Overall configured positive claim: PASS" in output
