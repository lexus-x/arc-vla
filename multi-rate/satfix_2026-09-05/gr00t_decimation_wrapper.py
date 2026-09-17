"""
DecimationSatfixWrapper: wraps a Gr00tPolicy so its emitted action chunk is decimated to
every k-th step and reconstructed via {zoh, tac_fold, tac_fold_satfix}, exactly mirroring
what evaluate_multirate_honest.py does for LIBERO/SmolVLA -- this is a pure post-hoc,
policy-agnostic transform on the (B,T,D) action chunk the model already produced, so it
applies to any BasePolicy without touching GR00T internals.

Per action key, treatment is decided from a hardcoded per-embodiment table, NOT from the
checkpoint's ActionConfig.rep -- for nvidia/GR00T-N1.6-3B's robocasa_panda_omron embodiment,
processor_config.json labels every action key "ABSOLUTE", but the actual RoboCasa robot
converter (external_dependencies/robocasa/robocasa/models/robots/__init__.py,
PandaOmronKeyConverter) shows the checkpoint's rep field is misleading here:
  - end_effector_position / end_effector_rotation: get_metadata() tags these
    "absolute: False" and unmap_action() feeds them straight into robosuite's OSC_POSE
    controller as per-step DELTA pose commands normalized to [-1,1] -- the exact
    saturating-controller pattern satfix targets in ManiSkill/LIBERO. These get full
    decimate+resample(+satfix) treatment.
  - gripper_close / control_mode: genuinely discrete (thresholded at 0.5 by unmap_action),
    not something to interpolate -> causal hold.
  - base_motion: not covered by get_metadata()'s explicit branches (falls to its
    "absolute: True" catch-all) -> held conservatively rather than assumed delta.

Note on the modality_configs attribute below: Gr00tSimPolicyWrapper (the wrapper this class
is meant to sit under) accesses `self.policy.modality_configs[...]` directly rather than
through get_modality_config(), so this class must expose that same plain attribute to be
wrapped by it.

Insertion point: gr00t/policy/policy.py's PolicyWrapper, same base class Gr00tSimPolicyWrapper
uses. Wrap the *inner* Gr00tPolicy with this before Gr00tSimPolicyWrapper so the flat-key
transform still happens last, e.g.:
    policy = Gr00tPolicy(...)
    policy = DecimationSatfixWrapper(policy, k=2, resampler="tac_fold_satfix")
    policy = Gr00tSimPolicyWrapper(policy)
"""
from typing import Any

import numpy as np

from gr00t.policy.policy import BasePolicy, PolicyWrapper

from resample_math import decimate_and_resample, RESAMPLERS

# Per-embodiment, per-action-key treatment: True = delta channel (clipped by a saturating
# per-step controller downstream -> decimate+resample+satfix applies), False = hold.
# Verified against PandaOmronKeyConverter.get_metadata/unmap_action, not the checkpoint's
# (misleading, all-ABSOLUTE) ActionConfig.rep. Add other embodiments here before reusing
# this wrapper for them -- an unlisted embodiment fails loudly rather than guessing.
EMBODIMENT_DELTA_KEYS: dict[str, dict[str, bool]] = {
    "robocasa_panda_omron": {
        "end_effector_position": True,
        "end_effector_rotation": True,
        "gripper_close": False,
        "control_mode": False,
        "base_motion": False,
    },
}


class DecimationSatfixWrapper(PolicyWrapper):
    def __init__(self, policy: BasePolicy, *, k: int, resampler: str, embodiment_tag: str, strict: bool = True):
        assert resampler in RESAMPLERS, f"unknown resampler {resampler!r}; choices={list(RESAMPLERS)}"
        assert embodiment_tag in EMBODIMENT_DELTA_KEYS, (
            f"no verified delta/hold table for embodiment {embodiment_tag!r}; "
            f"add one to EMBODIMENT_DELTA_KEYS before using this wrapper for it"
        )
        super().__init__(policy, strict=strict)
        self.k = k
        self.resampler = resampler
        self.delta_keys = EMBODIMENT_DELTA_KEYS[embodiment_tag]
        self.modality_configs = policy.get_modality_config()
        # diagnostic only: fraction of elements the unfixed tac_fold reconstruction would
        # have put outside [-1,1] before any satfix projection, per delta key. Populated
        # whenever resampler=="tac_fold_satfix" so the loss column is always reportable,
        # matching the ManiSkill/LIBERO harness's saturated_elems convention.
        self.unfixed_saturation: dict[str, list[float]] = {}

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def _get_action(self, observation, options=None):
        action, info = self.policy.get_action(observation, options)
        k = self.k
        out = {}
        for key, arr in action.items():
            is_delta = self.delta_keys.get(key, False)
            B, T, D = arr.shape
            if not is_delta or T < k:
                if is_delta or T < k:
                    out[key] = arr
                else:
                    n_blocks = T // k
                    held = np.repeat(arr[:, : n_blocks * k : k], k, axis=1)
                    if n_blocks * k < T:
                        held = np.concatenate([held, arr[:, n_blocks * k :]], axis=1)
                    out[key] = held.astype(arr.dtype)
                continue
            result = np.empty_like(arr)
            for b in range(B):
                result[b] = decimate_and_resample(arr[b], k, self.resampler)
            if self.resampler == "tac_fold_satfix":
                unfixed = np.empty_like(arr)
                for b in range(B):
                    unfixed[b] = decimate_and_resample(arr[b], k, "tac_fold")
                frac = float((np.abs(unfixed) > 1.0).mean())
                self.unfixed_saturation.setdefault(key, []).append(frac)
            out[key] = result.astype(arr.dtype)
        return out, info
