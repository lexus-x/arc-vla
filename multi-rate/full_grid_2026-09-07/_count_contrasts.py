"""Count paired contrasts stored in harness result JSONs."""
import json, glob

tot = sig = 0
files = 0
by_ref = {}
for pat in ("/home/user/Desktop/multi-rate/full_grid_2026-09-07/result_*.json",
            "/home/user/Desktop/multi-rate/seeds_2026-09-07/result_*.json"):
    for f in glob.glob(pat):
        if "_smoke" in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        files += 1
        for arm, cs in (d.get("contrasts") or {}).items():
            if not isinstance(cs, dict):
                continue
            for ref, c in cs.items():
                if isinstance(c, dict) and "p" in c:
                    tot += 1
                    by_ref[ref] = by_ref.get(ref, 0) + 1
                    if c["p"] < 0.05:
                        sig += 1
print(f"result files parsed: {files}")
print(f"total paired contrasts with p: {tot}")
print(f"nominal p<0.05 (no Holm here): {sig}")
print("by reference:", by_ref)
