#!/usr/bin/env python3
"""Harvest an expanded Push-T dataset (550 demos) from pusht_rl.h5,
reusing the existing 200 demos from demos_PushT-v1_200.npz to save time.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from harness import ManiSkillSim

TARGET_OUT = f"{HERE}/demos_PushT-v1_550.npz"
BASE_200 = f"{HERE}/demos_PushT-v1_200.npz"
TOTAL_DEMOS = 550

def main():
    if os.path.exists(TARGET_OUT):
        print(f"[skip] {TARGET_OUT} already exists.", flush=True)
        return

    sim = ManiSkillSim("PushT-v1")
    all_eligible = sim.eligible
    print(f"[data] Total eligible demos in pusht_rl.h5: {len(all_eligible)}", flush=True)
    assert len(all_eligible) >= TOTAL_DEMOS + 100, f"Need at least {TOTAL_DEMOS + 100} demos, got {len(all_eligible)}"

    O, A = [], []
    start_idx = 0
    if os.path.exists(BASE_200):
        print(f"[data] Reusing existing 200 demos from {BASE_200}...", flush=True)
        z = np.load(BASE_200, allow_pickle=True)
        O.extend(list(z["O"]))
        A.extend(list(z["A"]))
        start_idx = len(O)
        print(f"[data] Loaded {start_idx} cached demos.", flush=True)

    if start_idx < TOTAL_DEMOS:
        remaining_eps = all_eligible[start_idx:TOTAL_DEMOS]
        print(f"[harvest] Harvesting remaining {len(remaining_eps)} demos ({start_idx} to {TOTAL_DEMOS})...", flush=True)
        t0 = time.time()
        O_new, A_new = sim.harvest(remaining_eps)
        O.extend(O_new)
        A.extend(A_new)
        print(f"[harvest] Harvested {len(O_new)} demos in {time.time()-t0:.1f}s.", flush=True)

    sim.close()

    assert len(O) == TOTAL_DEMOS and len(A) == TOTAL_DEMOS
    tmp_out = f"{TARGET_OUT}.tmp.npz"
    np.savez(tmp_out, O=np.array(O, dtype=object), A=np.array(A, dtype=object))
    os.replace(tmp_out, TARGET_OUT)
    print(f"[DONE] Saved {TOTAL_DEMOS} demos to {TARGET_OUT} ({os.path.getsize(TARGET_OUT)/(1024*1024):.1f} MB)", flush=True)

if __name__ == "__main__":
    main()
