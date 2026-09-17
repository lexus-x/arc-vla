import glob, os, json
from safetensors import safe_open
base = "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH"
for f in sorted(glob.glob(base+"/**/checkpoints/8300/model.safetensors", recursive=True)):
    run = f.split("/runs/")[-1].split("/checkpoints")[0] if "/runs/" in f else f.replace(base,"")[:60]
    with safe_open(f, framework="pt") as h:
        ks = [k for k in h.keys() if "deeponet" in k or "head" in k or "trunk" in k or "branch" in k or "pool" in k]
        shapes = {k: list(h.get_slice(k).get_shape()) for k in ks}
    trunk0 = [v for k,v in shapes.items() if k.endswith("trunk.0.weight")]
    npool  = len({k.split(".pool")[0]+".pool"+k.split(".pool")[1].split(".")[0] for k in shapes if ".pool" in k})
    br     = [v for k,v in shapes.items() if "branch" in k and k.endswith(".0.weight")]
    print(f"{run:35s} nk={len(ks):3d} trunk0={trunk0} branch0={br} poolblk={npool}")
