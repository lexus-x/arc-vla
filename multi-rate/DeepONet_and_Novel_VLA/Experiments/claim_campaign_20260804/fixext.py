P = "mcgrid.py"
s = open(P).read()
old_start = s.index("def outcomes(v):")
old_end = s.index("def mcnemar(b, c):")
new = '''def outcomes(v):
    """-> {(task, seed): 0/1}. Seed is the pairing variable: the same seed is the same
    initial state, so a (task, seed) cell is genuinely paired across arms."""
    out = {}
    for t, blk in (v.get("per_task") or {}).items():
        for i, ep in enumerate(blk.get("episodes", [])):
            out[(str(t), ep.get("seed", i))] = int(bool(ep.get("success")))
    return out


def transforms(v):
    """The harness records how many times each rescale actually fired. A 40 Hz arm with a None
    or 0 counter silently ran unmodified -- this campaign has shipped that exact bug before."""
    d = v.get("diagnostics") or {}
    return {k: d.get(k) for k in ("spline_applied", "magscale_applied", "cadence", "anchor_applied")}


'''
s = s[:old_start] + new + s[old_end:]

# print the transform counters alongside each arm's rate
a = '        print(f"   {a:38s} n={len(o):4d}  {100*sum(o.values())/max(len(o),1):5.1f}%")'
assert s.count(a) == 1
s = s.replace(a, '        tr = transforms(v)\n'
                 '        fired = ", ".join(f"{k.replace(chr(95)+chr(97)+chr(112)+chr(112)+chr(108)+chr(105)+chr(101)+chr(100),chr(0))}" for k in [])\n'
                 '        note = " ".join(f"{k}={tr[k]}" for k in tr if tr[k] not in (None, 0))\n'
                 '        print(f"   {a:38s} n={len(o):4d}  {100*sum(o.values())/max(len(o),1):5.1f}%   {note}")', 1)
open(P, "w").write(s)
import ast; ast.parse(s)
print("EXTRACTOR PATCHED + SYNTAX OK")
