"""PRISM & ZOH Registered Runner for evaluate_multirate_honest.py."""
import sys
import os

sys.path.insert(0, "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804")
import evaluate_multirate_honest as E
from prism import PRISMAdapter

# Register ZOH and PRISM arms for v2_spatial
E.ARMS["v2_spatial_zoh_40env"] = ("v2_spatial", 40, 20, "hold")
E.ARMS["v2_spatial_prism_40env"] = ("v2_spatial", 40, 20, "prism")

# Hook resample_actions
prism_adapter = PRISMAdapter(source_rate_hz=20.0, target_rate_hz=40.0, has_gripper=True)
original_resample = getattr(E, "resample_actions", None)

def custom_resample(chunk, transform, target_hz, source_hz):
    if transform == "prism":
        return prism_adapter.resample_chunk(chunk, target_rate_hz=target_hz, source_rate_hz=source_hz)
    if original_resample is not None:
        return original_resample(chunk, transform, target_hz, source_hz)
    return chunk

E.resample_actions = custom_resample

if __name__ == "__main__":
    E.main()
