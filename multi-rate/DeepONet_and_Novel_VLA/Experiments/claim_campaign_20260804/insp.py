import json, glob, os
CK = "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/runs/m3_deeponet_s0/checkpoints/8300"
cfg = json.load(open(os.path.join(CK, "config.json")))
print("config keys:", list(cfg)[:25])
for k in cfg:
    if any(t in k.lower() for t in ("deeponet", "head", "_p", "fourier", "branch", "trunk", "d_model", "quer")):
        print("  ", k, "=", cfg[k])
import torch
from safetensors.torch import load_file
f = glob.glob(os.path.join(CK, "**", "*.safetensors"), recursive=True) + \
    glob.glob(os.path.join(CK, "**", "*.bin"), recursive=True)
print("weight files:", [os.path.basename(x) for x in f])
sd = load_file(f[0]) if f[0].endswith("safetensors") else torch.load(f[0], map_location="cpu")
for k, v in sd.items():
    if "branch" in k or "trunk" in k or "out" in k.lower() and "head" in k.lower():
        print(f"  {k:60s} {tuple(v.shape)}")
