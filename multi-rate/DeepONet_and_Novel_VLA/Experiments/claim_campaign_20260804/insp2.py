import os
from safetensors.torch import load_file
CK = "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/runs/m3_deeponet_s0/checkpoints/8300"
sd = load_file(os.path.join(CK, "model.safetensors"))
keys = [k for k in sd if "branch" in k or "trunk" in k or "pool" in k or "out_mlp" in k or "fourier" in k]
for k in sorted(keys):
    print(f"{k:62s} {tuple(sd[k].shape)}")
print("\ntotal params:", sum(v.numel() for v in sd.values()))
