"""Add the matched-budget 30K per-suite pairing: real DeepONet-v2 vs flow, Spatial/Object/Long.

Every replace asserts its match count -- a silent no-op patch has already cost this campaign one
full evaluation cycle.
"""
import re, sys

P = "evaluate_multirate_honest.py"
s = open(P).read()

# ---------------------------------------------------------------- MODELS
anchor = '    "flow_m1": ("flow", M1_CKPT, None, None),\n}'
assert s.count(anchor) == 1, f"MODELS anchor count={s.count(anchor)}"

V2 = '{PRIOR}/v2/deeponet_results'
FR = '{PRIOR}/paper_repro'
suites = [("spatial", "Spatial"), ("object", "Object"), ("long", "Long")]

lines = ['    "flow_m1": ("flow", M1_CKPT, None, None),',
         '    # --- matched-budget 30K per-suite pairing (added 2026-08-12) ---',
         '    # v2_* is the REAL DeepONetHeadV2: CrossAttnPool, p=256, n_fourier=16 (trunk_in=33).',
         '    # The older "don_v2" key points at m3_deeponet_s0, which is architecturally v1 --',
         '    # flat 960-dim branch, scalar-tau trunk, no pool. It cannot load into the v2 builder,',
         '    # which is exactly the exit=1 that killed don_v2_native_20env.',
         '    # Both sides are 30000 steps, so this pairing is budget-matched. Goal has no 30K',
         '    # checkpoint for EITHER model, so it is absent symmetrically, not dropped.']
for lo, Hi in suites:
    lines.append(f'    "v2_{lo}": ("deeponet", f"{V2}/{Hi}/runs/m3_s0/checkpoints/30000", "deeponet", 16),')
for lo, Hi in suites:
    lines.append(f'    "flow30_{lo}": ("flow", f"{FR}/{Hi}/runs/flow_s0/checkpoints/30000", None, None),')
lines.append('}')
s = s.replace(anchor, "\n".join(lines), 1)

# ---------------------------------------------------------------- MODEL_DATASET
danchor = '    "asrc_long":   "lerobot/libero_10_image",\n}'
assert s.count(danchor) == 1, f"MODEL_DATASET anchor count={s.count(danchor)}"
dl = ['    "asrc_long":   "lerobot/libero_10_image",',
      '    # 30K per-suite checkpoints were TRAINED on their own suite, so their normalization',
      '    # stats must come from that suite -- not from Spatial. Spatial falls back to DEFAULT.',
      '    "v2_object":     "lerobot/libero_object_image",',
      '    "v2_long":       "lerobot/libero_10_image",',
      '    "flow30_object": "lerobot/libero_object_image",',
      '    "flow30_long":   "lerobot/libero_10_image",',
      '}']
s = s.replace(danchor, "\n".join(dl), 1)

# ---------------------------------------------------------------- ARMS
aanchor = '    "flow_m1_cadence_40env":       ("flow_m1", 40, None, "magscale+cadence"),'
assert s.count(aanchor) == 1, f"ARMS anchor count={s.count(aanchor)}"
al = [aanchor,
      '    # --- the 2x2 that is actually runnable: {v2, flow} x {native 20 Hz, spline 40 Hz} ---',
      '    # FOLDING IS NOT IN THIS GRID AND CANNOT BE. Folding integrates a velocity field at a',
      '    # chosen dt; neither v2 nor flow has a velocity field -- both emit a chunk directly.',
      '    # There is no post-hoc folding operator, so "v2+folding" and "flow+folding" are not',
      '    # missing runs, they are undefined. Cubic spline is the best either can do off-native.']
for lo, _ in suites:
    al.append(f'    "v2_{lo}_native_20env":            ("v2_{lo}", 20, 20, None),')
    al.append(f'    "v2_{lo}_cadmag_spline_40env":     ("v2_{lo}", 40, 20, "spline+magscale+cadence"),')
for lo, _ in suites:
    al.append(f'    "flow30_{lo}_native_20env":        ("flow30_{lo}", 20, None, None),')
    al.append(f'    "flow30_{lo}_cadmag_spline_40env": ("flow30_{lo}", 40, None, "spline+magscale+cadence"),')
s = s.replace(aanchor, "\n".join(al), 1)

assert s != open(P).read(), "patch produced no change"
open(P, "w").write(s)

import ast
ast.parse(s)
print("PATCH OK + SYNTAX OK")

import importlib, os
os.environ.setdefault("DEEPONET_HEAD", "asrc")
m = importlib.import_module("evaluate_multirate_honest")
new = [a for a in m.ARMS if a.startswith(("v2_", "flow30_"))]
print(f"{len(new)} new arms:")
for a in new:
    mk = m.ARMS[a][0]
    ck = m.MODELS[mk][1]
    print(f"  {a:36s} ckpt_exists={os.path.isdir(ck)}  ds={m.MODEL_DATASET.get(mk, m.DEFAULT_DATASET)}")
