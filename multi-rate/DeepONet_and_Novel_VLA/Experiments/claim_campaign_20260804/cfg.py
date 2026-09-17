import json
B = "/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH"
for tag, p in (("v2   Spatial", f"{B}/v2/deeponet_results/Spatial/runs/m3_s0/checkpoints/30000/config.json"),
               ("flow Spatial", f"{B}/paper_repro/Spatial/runs/flow_s0/checkpoints/30000/config.json")):
    c = json.load(open(p))
    keys = ("chunk_size", "n_action_steps", "n_obs_steps", "num_steps", "max_action_dim")
    print(f"{tag}: " + "  ".join(f"{k}={c.get(k)}" for k in keys))
