"""Download and organize ManiSkill 3 task demonstrations."""
import sys
from mani_skill.utils.download_demo import parse_args, main

TASKS = [
    "StackCube-v1",
    "PlugCharger-v1",
    "PullCube-v1",
    "LiftPegUpright-v1",
    "PokeCube-v1",
    "RollBall-v1",
    "PushCube-v1",
]

out_dir = "/home/user/maniskill_data/raw"
for t in TASKS:
    print(f"[+] Downloading {t}...")
    try:
        args = parse_args(["-o", out_dir, t])
        main(args)
    except Exception as e:
        print(f"[-] Error downloading {t}: {e}")

print("[+] Done downloading all tasks.")
