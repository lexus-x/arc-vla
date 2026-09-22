import os, hashlib, json

ckpts = {
    "RoboMimic Lift": "dp_lift.pt",
    "RoboMimic Can": "dp_can.pt",
    "RoboMimic Square": "dp_square.pt",
    "Push-T": "dp_PushT-v1.pt"
}

demos = {
    "RoboMimic Lift": "demos_lift_200.npz (200 demos, robomimic v0.3 ph dataset)",
    "RoboMimic Can": "demos_can_200.npz (200 demos, robomimic v0.3 ph dataset)",
    "RoboMimic Square": "demos_square_200.npz (200 demos, robomimic v0.3 ph dataset)",
    "Push-T": "demos_PushT-v1_200.npz (200 demos, 10Hz native)"
}

print("=== CHECKPOINT PROVENANCE ===")
for name, fname in ckpts.items():
    p = os.path.abspath(fname)
    with open(p, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    size_mb = os.path.getsize(p) / (1024*1024)
    print(f"Name: {name}")
    print(f"  File: {p}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  SHA-256: {sha256}")
    print(f"  Source Demos: {demos[name]}")
    print(f"  Eval Protocol: n=50 heldout episodes, multi-worker simulation")
    print()
