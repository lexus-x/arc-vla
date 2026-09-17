import os
from safetensors.torch import load_file
B = "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH"
for tag, ck in [("v2_spatial", f"{B}/v2/deeponet_results/Spatial/runs/m3_s0/checkpoints/30000"),
                ("flow30_spatial", f"{B}/paper_repro/Spatial/runs/flow_s0/checkpoints/30000")]:
    sd = load_file(os.path.join(ck, "model.safetensors"))
    print(f"=== {tag} ===")
    for k in sorted(sd):
        if any(t in k for t in ("branch", "trunk", "out_mlp", "pool.", "fourier", "freq")):
            print(f"   {k:58s} {tuple(sd[k].shape)}")
    print()
