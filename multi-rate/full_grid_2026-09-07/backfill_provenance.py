"""One-off provenance backfill for existing harness result JSON files."""
from collections import Counter
from pathlib import Path
import glob
import json
import re
import shutil


BACKUP_DIR = Path("_pre_backfill_backup")


def inferred_provenance(filename):
    """Infer the result provenance encoded by harness.py's output filename."""
    head = "bspline" if "_bspline" in filename else "blocksum" if "_blocksum" in filename else "step"
    seed = re.search(r"_s(\d+)", filename)
    fold = re.search(r"_f(\d+)", filename)
    return {
        "head": head,
        "seed": int(seed.group(1)) if seed else 0,
        "fold": int(fold.group(1)) if fold else -1,
    }


def main():
    result_files = [Path(path) for path in sorted(glob.glob("result_*.json"))]
    if BACKUP_DIR.exists():
        raise SystemExit(f"refusing to overwrite existing backup directory: {BACKUP_DIR}")

    # Preserve every input before any result file can be modified.
    BACKUP_DIR.mkdir()
    for path in result_files:
        shutil.copy2(path, BACKUP_DIR / path.name)

    backfilled = 0
    heads, seeds, folds = Counter(), Counter(), Counter()
    for path in result_files:
        with path.open() as fh:
            result = json.load(fh)
        added = False
        for field, value in inferred_provenance(path.name).items():
            if field not in result:
                result[field] = value
                added = True
        if added:
            with path.open("w") as fh:
                json.dump(result, fh, indent=2)
                fh.write("\n")
            backfilled += 1
        heads[result["head"]] += 1
        seeds[result["seed"]] += 1
        folds[result["fold"]] += 1

    print(f"backfilled files: {backfilled}")
    print(f"head counts: {dict(sorted(heads.items()))}")
    print(f"seed counts: {dict(sorted(seeds.items()))}")
    print(f"fold counts: {dict(sorted(folds.items()))}")


if __name__ == "__main__":
    main()
