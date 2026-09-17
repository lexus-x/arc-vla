"""Fix UnboundLocalError: `heads` is bound only for ti/til/asrc, but the spline block guards on
`backbone == "deeponet"`, which is also true for the v2 head -- so every v2 spline arm dies at
startup.

DeepONetHeadV2 has NO action_offset/action_scale buffers (grep: only RateIntegratedDeepONetHead
registers them, and only because RK4 divides by action_scale mid-integration). So for v2 there is
no head-side stat to compare the dataset stats against, and the correct behaviour is to skip the
comparison -- same as the flow path -- not to invent one.

Also hardens the anchor block: with `heads` empty its for-loop silently does nothing, which would
produce an "anchor" arm that never anchored. This campaign has already shipped two silent no-op
arms; make it fatal instead.
"""
P = "evaluate_multirate_honest.py"
s = open(P).read()

# 1) bind `heads` on every path
a1 = "    policy, (preprocessor, postprocessor) = screen.load_policy(backbone, checkpoint, stats)\n"
assert s.count(a1) == 1, f"load_policy anchor count={s.count(a1)}"
s = s.replace(a1, a1 + "    # Rate-integrated heads only. DeepONetHeadV2 and the flow backbone have none, so this\n"
                       "    # stays empty for them and every `heads` consumer below must handle that.\n"
                       "    heads = []\n", 1)

# 2) spline stat check: only meaningful when a rate-integrated head actually holds the stats
a2 = '        if backbone == "deeponet":   # must equal the head\'s own stats, or every prior arm shifts\n'
assert s.count(a2) == 1, f"spline guard count={s.count(a2)}"
s = s.replace(a2, '        # Only RateIntegratedDeepONetHead stores action stats and uses them inside RK4, so\n'
                  '        # only there can a dataset/head mismatch change the dynamics. v2 and flow normalize\n'
                  '        # via the pre/postprocessor, so there is nothing to cross-check.\n'
                  '        if backbone == "deeponet" and heads:   # must equal the head\'s own stats\n', 1)

# 3) anchor: empty `heads` would make the hook loop a silent no-op
a3 = '        anc = {"n": 0}\n        for _h in heads:\n'
assert s.count(a3) == 1, f"anchor loop count={s.count(a3)}"
s = s.replace(a3, '        if not heads:\n'
                  '            raise SystemExit(f"[FATAL] {arm}: anchor needs a RateIntegratedDeepONetHead; "\n'
                  '                             f"this checkpoint has none, the hook would be a no-op")\n'
                  '        anc = {"n": 0}\n        for _h in heads:\n', 1)

open(P, "w").write(s)
import ast; ast.parse(s)
print("PATCH OK + SYNTAX OK")
print(f"heads=[] inserted: {s.count('    heads = []') == 1}")
print(f"guarded spline    : {s.count('backbone == ' + chr(34) + 'deeponet' + chr(34) + ' and heads') == 1}")
