"""State history must follow the HEAD, not the backbone.

`DEEPONET_STATE_HISTORY_STEPS=8` was applied to every deeponet backbone. That is right for
ti/til/asrc (and ncde_style), which consume eight chronological states. It is WRONG for the v2
head: modeling_smolvla_deeponet_v2.py sets `state_history_steps = 8 if deeponet_head ==
"ncde_style" else None`, and the v2 checkpoints carry `n_obs_steps=1`. Forcing 8 makes
evaluate_height_screen call configure_state_history(8), so v2 is fed eight stacked states it was
never trained on.

It fails silently because state history changes no `deeponet.*` weight shape, so load_policy's
tensor guard passes. Measured cost: v2 Spatial at the original protocol (replan=5, 520 steps)
scored 31.0% here vs the original harness's 85.0%.
"""
P = "evaluate_multirate_honest.py"
s = open(P).read()

a = '        os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"\n'
assert s.count(a) == 1, f"state-history anchor count={s.count(a)}"
new = ('        # Eight chronological states belong to the rate-integrated / ncde heads only. The v2\n'
       '        # head declares state_history_steps=None and its checkpoints carry n_obs_steps=1;\n'
       '        # forcing 8 there feeds it a state tensor it never saw in training, costs ~54 pp,\n'
       '        # and passes the tensor guard because no deeponet.* shape changes.\n'
       '        os.environ["DEEPONET_STATE_HISTORY_STEPS"] = (\n'
       '            "8" if deeponet_head in {"ti", "til", "asrc", "ncde_style"} else "1")\n')
s = s.replace(a, new, 1)

open(P, "w").write(s)
import ast; ast.parse(s)
print("PATCH OK + SYNTAX OK")

import os, importlib
os.environ.setdefault("DEEPONET_HEAD", "asrc")
m = importlib.import_module("evaluate_multirate_honest")
for k in ("asrc", "v2_spatial", "flow30_spatial"):
    if k in m.MODELS:
        print(f"  {k:14s} head={m.MODELS[k][2]}")
