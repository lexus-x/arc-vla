"""Classify every checkpoint by action-head architecture. v2 == has CrossAttnPool (pool.*)."""
import os, glob
from safetensors.torch import load_file

ROOT = "/home/user/DeepONet_and_Novel_VLA"
files = glob.glob(os.path.join(ROOT, "**", "model.safetensors"), recursive=True)
print(f"{len(files)} checkpoints\n")
rows = []
for f in files:
    try:
        sd = load_file(f)
    except Exception as e:
        rows.append((f, "UNREADABLE", str(e)[:40])); continue
    k = list(sd)
    has_pool = any(".pool." in x for x in k)
    has_don = any("deeponet" in x for x in k)
    has_rhs = any(".rhs." in x for x in k)
    has_flow = any("action_out_proj" in x or "denoise" in x or "time_mlp" in x for x in k)
    if has_rhs:      kind = "RATE-INTEGRATED (ti/til/asrc)"
    elif has_pool:   kind = "DeepONet **v2** (CrossAttnPool)"
    elif has_don:    kind = "DeepONet v1 (flat branch)"
    elif has_flow:   kind = "flow"
    else:            kind = "other"
    bshape = next((tuple(sd[x].shape) for x in k if x.endswith("branch.0.weight")), None)
    rows.append((f.replace(ROOT + "/", ""), kind, f"branch0={bshape}"))
for f, kind, extra in sorted(rows, key=lambda r: r[1]):
    print(f"{kind:32s} {extra:26s} {f}")
