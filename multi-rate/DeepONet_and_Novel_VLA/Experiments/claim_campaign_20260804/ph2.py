P = "evaluate_multirate_honest.py"
s = open(P).read()
a = ('        if backbone == "deeponet":\n'
     '            if not torch.allclose(off.cpu(), heads[0].action_offset[:POSE_DIMS].detach().float().cpu(), atol=1e-6):\n')
assert s.count(a) == 1, f"magscale guard count={s.count(a)}"
s = s.replace(a, ('        # Same reasoning as the spline block: only a rate-integrated head carries these\n'
                  '        # stats internally. v2/flow normalize via the pre/postprocessor, nothing to check.\n'
                  '        if backbone == "deeponet" and heads:\n'
                  '            if not torch.allclose(off.cpu(), heads[0].action_offset[:POSE_DIMS].detach().float().cpu(), atol=1e-6):\n'), 1)
open(P, "w").write(s)
import ast; ast.parse(s)
import re
bad = [ln for ln in s.splitlines() if "heads[0]" in ln]
print("PATCH OK + SYNTAX OK")
print(f"remaining heads[0] uses: {len(bad)} (all must sit inside the ti/til/asrc block)")
