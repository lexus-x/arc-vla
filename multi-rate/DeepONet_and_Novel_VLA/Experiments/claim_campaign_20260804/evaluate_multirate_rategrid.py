"""Multi-rate evaluation at a genuine LIBERO control_freq.

Replaces evaluate_spline_folding_matrix.py, which could not measure what it claimed:

  1. The env was always 20 Hz -- LiberoEnv._make_envs_task never passes `control_freq`
     into OffScreenRenderEnv, so every "40 Hz" arm ran in the stock 20 Hz sim.
  2. REPLAN = 1 pinned n_action_steps to 1, so only chunk[0] was ever consumed.
     CubicSpline/np.interp pass exactly through their knots, so the spline and linear
     arms were provably inert -- they reported unmodified performance.
  3. MAX_STEPS was fixed at 220 regardless of rate, so a head integrated at dt=1/40
     moved at half speed into a budget calibrated for full speed and got truncated.
  4. folding/enhanced_folding were configured behind silent `if heads:` and
     `hasattr(..., "set_enhanced_folding")` guards, so a miss was a no-op, not an error.

Here every rate-dependent quantity is derived from the arm's env_freq, so the arms are
matched in wall-clock time rather than in env steps:

    steps  = WALLCLOCK_S * env_freq    (11.0 s -> 220 @ 20 Hz, 440 @ 40 Hz)
    replan = REPLAN_S    * env_freq    ( 0.5 s ->  10 @ 20 Hz,  20 @ 40 Hz)

Because REPLAN > 1 here, none of the numbers this produces are comparable to any
earlier run in this campaign -- those all executed a single action per plan.

Nothing is configured silently: if an intervention cannot be applied, this aborts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline

# Matched-budget protocol. 220 steps at 20 Hz is the campaign's historical budget.
WALLCLOCK_S = 11.0

# Per-suite episode budget. A single 11.0 s budget is the LIBERO-Spatial figure; applied to the
# other suites it TRUNCATES them -- every rollout in suite_libero_{object,goal,10}.log ended at
# the cap (220/220 @20Hz, 330/330 @30Hz) with ZERO early terminations, which is a horizon
# signature, not a capability signature. Values below are the standard LIBERO max_steps
# convention expressed as wall-clock at NATIVE_HZ=20: spatial 220, object 280, goal 300,
# long 520 steps. Spatial is UNCHANGED so all existing results are preserved exactly.
SUITE_WALLCLOCK = {
    "libero_spatial": 11.0,
    "libero_object":  14.0,
    "libero_goal":    15.0,
    "libero_10":      26.0,
}
WALLCLOCK_OVERRIDE = None  # set by --wallclock; None => use SUITE_WALLCLOCK
REPLAN_S = 0.5
REPLAN_OVERRIDE_STEPS = None  # set by --replan_steps; None => REPLAN_S * env_freq
NATIVE_HZ = 20.0
HORIZON_S = 2.5  # RateIntegratedDeepONetHead.horizon_s; chunk is ceil(HORIZON_S * rate) steps
SETTLE_STEPS_20HZ = 10  # LiberoEnv.num_steps_wait default, scaled per rate to fix settle time
N_TASKS = 10
POSE_DIMS = 6  # dims 0:6 are integrated pose deltas; 6+ is the direct gripper path

CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
DEEPONET_CKPT = f"{CAMPAIGN}/runs/til_s0/checkpoints/8300"
ASRC_CKPT = f"{CAMPAIGN}/runs/asrc_s0/checkpoints/8300"
FLOW_CKPT = ("/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH"
             "/paper_repro/Spatial/runs/flow_s0/checkpoints/30000")

PRIOR = ("/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH")
M3_CKPT = f"{PRIOR}/runs/m3_deeponet_s0/checkpoints/8300"

# Normalization stats MUST come from the dataset a checkpoint was TRAINED on, not from the suite
# being evaluated. rate_integrated_deeponet.py:130 uses action_scale inside the RK4 integration,
# so a mismatch changes the head's dynamics rather than merely rescaling its output.
# Anything absent falls back to Spatial, which is correct for every pre-2026-08-11 checkpoint.
DEFAULT_DATASET = "lerobot/libero_spatial_image"
MODEL_DATASET = {
    "asrc_object": "lerobot/libero_object_image",
    "asrc_goal":   "lerobot/libero_goal_image",
    "asrc_long":   "lerobot/libero_10_image",
    # 30K per-suite checkpoints were TRAINED on their own suite, so their normalization
    # stats must come from that suite -- not from Spatial. Spatial falls back to DEFAULT.
    "v2_object":     "lerobot/libero_object_image",
    "v2_long":       "lerobot/libero_10_image",
    "flow30_object": "lerobot/libero_object_image",
    "flow30_long":   "lerobot/libero_10_image",
}      # don_v2 head, matched 8.3K
M1_CKPT = f"{PRIOR}/runs/m1_flow_s0/checkpoints/8300"          # flow baseline, matched 8.3K

# model key -> (kind, checkpoint, DEEPONET_HEAD, DEEPONET_FOURIER)
#   The env vars must match what the checkpoint was trained with or load_policy's tensor
#   shape check aborts. til_s0 trained with fourier=16; asrc is capped at 6 because
#   consistency_rates includes 5 Hz (n_fourier <= floor(5 * 2.5 / 2)).
#   Only asrc trained with a cross-rate consistency loss, so only asrc can be expected
#   to re-integrate at an off-native rate -- see gate_rate_invariance.py.
MODELS = {
    "til": ("deeponet", DEEPONET_CKPT, "til", 16),
    "asrc": ("deeponet", ASRC_CKPT, "asrc", 6),
    "flow": ("flow", FLOW_CKPT, None, None),
    "don_v2": ("deeponet", M3_CKPT, "deeponet", 0),
    # Per-suite asrc checkpoints, 8300 steps, budget-matched to asrc_s0. Trained 2026-08-11
    # because asrc_s0 scores 0/50 on object/goal/10 even at the full standard horizon.
    "asrc_object": ("deeponet", f"{CAMPAIGN}/runs/asrc_object_s0/checkpoints/8300", "asrc", 6),
    "asrc_goal":   ("deeponet", f"{CAMPAIGN}/runs/asrc_goal_s0/checkpoints/8300",   "asrc", 6),
    "asrc_long":   ("deeponet", f"{CAMPAIGN}/runs/asrc_long_s0/checkpoints/8300",   "asrc", 6),
    "flow_m1": ("flow", M1_CKPT, None, None),
    # --- matched-budget 30K per-suite pairing (added 2026-08-12) ---
    # v2_* is the REAL DeepONetHeadV2: CrossAttnPool, p=256, n_fourier=16 (trunk_in=33).
    # The older "don_v2" key points at m3_deeponet_s0, which is architecturally v1 --
    # flat 960-dim branch, scalar-tau trunk, no pool. It cannot load into the v2 builder,
    # which is exactly the exit=1 that killed don_v2_native_20env.
    # Both sides are 30000 steps, so this pairing is budget-matched. Goal has no 30K
    # checkpoint for EITHER model, so it is absent symmetrically, not dropped.
    "v2_spatial": ("deeponet", f"{PRIOR}/v2/deeponet_results/Spatial/runs/m3_s0/checkpoints/30000", "deeponet", 16),
    "v2_object": ("deeponet", f"{PRIOR}/v2/deeponet_results/Object/runs/m3_s0/checkpoints/30000", "deeponet", 16),
    "v2_long": ("deeponet", f"{PRIOR}/v2/deeponet_results/Long/runs/m3_s0/checkpoints/30000", "deeponet", 16),
    "flow30_spatial": ("flow", f"{PRIOR}/paper_repro/Spatial/runs/flow_s0/checkpoints/30000", None, None),
    "flow30_object": ("flow", f"{PRIOR}/paper_repro/Object/runs/flow_s0/checkpoints/30000", None, None),
    "flow30_long": ("flow", f"{PRIOR}/paper_repro/Long/runs/flow_s0/checkpoints/30000", None, None),
    # proof40: asrc fine-tuned at 40 Hz (consistency_rates=40, 2000 head-only steps)
    "asrc40ft": ("deeponet", "/media/user/C2FE578FFE577A9D/claim_campaign_proof40/runs/asrc40hz_ft_s0_r2/checkpoints/2000", "asrc", 6),
    # --- rate grid, added 2026-08-29. Local FT checkpoints (NOT the external _r2).
    # All fine-tuned from asrc_s0/8300 on the SAME 20 Hz dataset; only
    # consistency_rates differs (10 / 40 / "10,40"). Not rate-specific TRAINING.
    "asrc10ft":  ("deeponet", f"{CAMPAIGN}/runs/asrc10hz_ft_s0/checkpoints/2000", "asrc", 6),
    "asrc40ftL": ("deeponet", f"{CAMPAIGN}/runs/asrc40hz_ft_s0/checkpoints/2000", "asrc", 6),
    "asrccombo": ("deeponet", f"{CAMPAIGN}/runs/asrc_combo_ft_s0/checkpoints/4000", "asrc", 6),
}

# arm -> (model key, env_freq, head_rate, chunk transform)
#   head_rate is the rate the DeepONet re-integrates its ODE at (None => leave native).
#   "naive" arms feed a 20 Hz-native chunk into a faster env; "folding" arms re-integrate
#   the ODE at the env rate; "spline" arms resample and rescale the 20 Hz chunk.
ARMS = {
    "til_native_20env":  ("til", 20, 20, None),
    "til_naive_40env":   ("til", 40, 20, None),
    "til_folding_40env": ("til", 40, 40, None),
    "til_spline_40env":  ("til", 40, 20, "spline"),
    # Zero-order hold: query the policy only every env_freq/20 steps and repeat the action.
    # The boring baseline, and the only one that fixes OBSERVATION CADENCE -- the 8-step state
    # history then spans 0.4 s as at 20 Hz, instead of 0.2 s. Tests the one hypothesis the
    # end-effector traces left standing: naive_40env already travels 0.78-1.05x the reference
    # and still scores 0/50, so per-step magnitude is not what breaks 40 Hz.
    "til_zoh_40env":     ("til", 40, 20, "hold"),
    # Matched-cadence arms: observation cadence identical to ZOH (policy queried every
    # env_freq/20 steps, so the state history spans 0.4 s), but the intermediate action is
    # RESOLVED instead of held. ZOH repeats a stale action for half the interval; these fill it
    # from the chunk. The only difference between the two is where the intermediate comes from:
    #   cadence          -> the operator's own 40 Hz re-integration  (the DeepONet's claim)
    #   spline+cadence   -> cubic-spline interpolation of the 20 Hz chunk (fair competitor)
    # This is the only comparison in which the operator head has a principled edge over ZOH.
    "til_cadence_folding_40env": ("til", 40, 40, "cadence"),
    "til_cadence_spline_40env":  ("til", 40, 20, "spline+cadence"),
    # Magnitude-corrected cadence. Both arms above halve the per-step delta and so under-travel;
    # magscale restores native travel. These two are matched on cadence AND travel rate, so the
    # ONLY remaining difference is how the intermediate action is computed:
    #   operator re-integration vs cubic-spline interpolation. This is the fair head-to-head.
    "til_cadmag_folding_40env": ("til", 40, 40, "magscale+cadence"),
    "til_cadmag_spline_40env":  ("til", 40, 20, "spline+magscale+cadence"),
    # 30 Hz: ratio 1.5. The only regime with STRUCTURAL headroom for a continuous-time head --
    # integer zero-order hold cannot express a 1.5x ratio, and cubic spline must guess between
    # 20 Hz samples, whereas the operator evaluates its trajectory exactly at the 30 Hz grid.
    # zoh_30env is the honest cheap control: non-uniform hold (1,2,1,2,... env steps).
    "til_cadmag_folding_30env": ("til", 30, 30, "magscale+cadence"),
    "til_cadmag_spline_30env":  ("til", 30, 20, "spline+magscale+cadence"),
    "til_zoh_30env":            ("til", 30, 20, "hold"),
    # 10 Hz: SUB-native, the one regime with a mechanism that favours integration over
    # interpolation. Below the native rate the chunk must be DOWNsampled: cubic spline
    # point-samples its interpolant at 10 Hz and therefore aliases any trajectory content above
    # the new 5 Hz Nyquist, while the operator integrates the ODE over each 0.1 s step and so
    # returns the step AVERAGE. Averaging vs point-sampling is a real asymmetry, unlike the 1%
    # interpolation error that bounds the 30/40 Hz arms.
    # NOTE: no cadence arms here. The phase accumulator advances >=1 per env step when
    # env_freq <= NATIVE_HZ, so every step is a fresh query and no intermediate is ever
    # resolved -- the harness FATALs on that, correctly. Observation cadence is 10 Hz for BOTH
    # arms, so it is not a confound between them.
    # magscale direction is UNVERIFIED at sub-native rates: k = 10/20 = 0.5 halves the per-step
    # delta, but each control step lasts twice as long so the OSC has twice as long to converge.
    # The 40 Hz calibration may not extrapolate, so both with and without are run and the data
    # decides.
    "asrc_folding_10env":     ("asrc", 10, 10, None),
    "asrc_spline_10env":      ("asrc", 10, 20, "spline"),
    "asrc_mag_folding_10env": ("asrc", 10, 10, "magscale"),
    "asrc_mag_spline_10env":  ("asrc", 10, 20, "spline+magscale"),
    # ASRC arms. til_s0 trained ONLY at 20 Hz, so its off-rate integration is out of
    # distribution, while spline resamples the in-distribution 20 Hz chunk and inherits
    # trained-quality trajectory for free. That is the gap. asrc_s0 was trained with a
    # cross-rate consistency loss at rates (5,10,25,40,50).
    # PRE-REGISTERED PREDICTION: 40 Hz IS in that set and 30 Hz is NOT, so asrc should close
    # the gap at 40 Hz and not at 30 Hz. If it closes at both, the mechanism is not what is
    # helping; if it closes at neither, cross-rate consistency does not transfer to control.
    "asrc_native_20env":        ("asrc", 20, 20, None),
    # --- Cell D v2 (cellD_v2_closedloop_prereg.json). env 20 / head 20 so the cadence divisor
    # and magscale are never invoked; all rate manipulation happens inside the chunk transform,
    # where the inertness gate can see it. Decode the SAME decimated stream three ways.
    "asrc_dec2_tacfold_20env": ("asrc", 20, 20, "decimate2+tac_fold"),
    "asrc_dec2_spline_20env":  ("asrc", 20, 20, "decimate2+spline_cum"),
    "asrc_dec2_zoh_20env":     ("asrc", 20, 20, "decimate2+zoh_fold"),
    # Deliberately inert control: k=1 + zoh_fold is an identity transform. Used ONCE to prove
    # the inertness gate actually fires; must never appear in a contrast.
    "asrc_inert_probe_20env":  ("asrc", 20, 20, "decimate1+zoh_fold"),
    # --- Cross-suite Cell D extension (2026-08-29): the same k=2 decode contrast on the
    # other three LIBERO suites, each with its OWN suite-trained 8300-step checkpoint.
    # native arms included so a both-arms-collapsed cell is VOID, not mistaken for a tie.
    "asrc_object_native_20env":       ("asrc_object", 20, 20, None),
    "asrc_object_dec2_tacfold_20env": ("asrc_object", 20, 20, "decimate2+tac_fold"),
    "asrc_object_dec2_spline_20env":  ("asrc_object", 20, 20, "decimate2+spline_cum"),
    "asrc_object_dec2_zoh_20env":     ("asrc_object", 20, 20, "decimate2+zoh_fold"),
    "asrc_goal_native_20env":       ("asrc_goal", 20, 20, None),
    "asrc_goal_dec2_tacfold_20env": ("asrc_goal", 20, 20, "decimate2+tac_fold"),
    "asrc_goal_dec2_spline_20env":  ("asrc_goal", 20, 20, "decimate2+spline_cum"),
    "asrc_goal_dec2_zoh_20env":     ("asrc_goal", 20, 20, "decimate2+zoh_fold"),
    "asrc_long_native_20env":       ("asrc_long", 20, 20, None),
    "asrc_long_dec2_tacfold_20env": ("asrc_long", 20, 20, "decimate2+tac_fold"),
    "asrc_long_dec2_spline_20env":  ("asrc_long", 20, 20, "decimate2+spline_cum"),
    "asrc_long_dec2_zoh_20env":     ("asrc_long", 20, 20, "decimate2+zoh_fold"),
    "asrc_cadmag_folding_40env": ("asrc", 40, 40, "magscale+cadence"),

    # --- 3x3 RATE GRID (2026-08-29) -------------------------------------------
    # rows: generalist (asrc_s0, consistency_rates 5,10,25,40,50) vs a model
    # fine-tuned toward 10 Hz vs one fine-tuned toward 40 Hz.
    # cols: served at 10 / 20 / 40 Hz. Transform per rate is held FIXED across
    # rows so the only thing varying down a column is the checkpoint.
    "rategrid_base_10env": ("asrc", 10, 10, "magscale"),
    "rategrid_base_20env": ("asrc", 20, 20, None),
    "rategrid_base_40env": ("asrc", 40, 40, "magscale+cadence"),
    "rategrid_ft10_10env": ("asrc10ft", 10, 10, "magscale"),
    "rategrid_ft10_20env": ("asrc10ft", 20, 20, None),
    "rategrid_ft10_40env": ("asrc10ft", 40, 40, "magscale+cadence"),
    "rategrid_ft40_10env": ("asrc40ftL", 10, 10, "magscale"),
    "rategrid_ft40_20env": ("asrc40ftL", 20, 20, None),
    "rategrid_ft40_40env": ("asrc40ftL", 40, 40, "magscale+cadence"),
    "asrc_cadmag_spline_40env":  ("asrc", 40, 20, "spline+magscale+cadence"),
    # Parameter-free zero-order hold at 40 Hz -- the baseline the operator
    # claim has to beat. Mirrors til_zoh_40env: head stays at native 20 Hz,
    # policy queried every env_freq/20 steps, action repeated in between.
    "asrc_zoh_40env":            ("asrc", 40, 20, "hold"),
    "asrc_cadmag_folding_30env": ("asrc", 30, 30, "magscale+cadence"),
    # RATE-ANCHORED INTEGRATION (RAI) -- the mechanism this campaign is testing.
    # folding pushes the trunk rate token to log2(env_freq/20) != 0, which is off-distribution;
    # spline keeps the token at 20 and inherits a trained-quality trajectory for free. That
    # asymmetry is the measured gap. RAI removes it: integrate with dt = 1/env_freq so the
    # trajectory is resolved exactly on the target grid (no interpolation at all), while the
    # trunk is queried at the NATIVE rate token it was trained on. Spline cannot do this (it has
    # no field to query) and folding cannot (its query is off-distribution).
    "asrc_anchor_30env": ("asrc", 30, 30, "anchor+magscale+cadence"),
    "asrc_anchor_40env": ("asrc", 40, 40, "anchor+magscale+cadence"),
    "asrc_cadmag_spline_30env":  ("asrc", 30, 20, "spline+magscale+cadence"),
    # Unmatched asrc arms (no cadence/magscale), kept for the ablation record: these are the
    # configurations that scored 0/50 on til and are expected to score 0 here too.
    "asrc_naive_40env":   ("asrc", 40, 20, None),
    "asrc_folding_40env": ("asrc", 40, 40, None),
    # 2x2 ablation of the two harness fixes, folding decode, 40 Hz.
    #   asrc_folding_40env        = neither
    #   asrc_cadence_folding_40env= cadence only
    #   asrc_mag_folding_40env    = magnitude only
    #   asrc_cadmag_folding_40env = both
    "asrc_cadence_folding_40env": ("asrc", 40, 40, "cadence"),
    "asrc_mag_folding_40env":     ("asrc", 40, 40, "magscale"),
    "asrc_spline_40env":  ("asrc", 40, 20, "spline"),
    "don_v2_native_20env":  ("don_v2", 20, 20, None),
    # don_v2 has NO rate input (plain DeepONetHeadV2: forward(prefix, pad_mask)), so head_rate is
    # pinned at 20 and its only off-native decodes are harness-side resamplers. These are therefore
    # BASELINE arms, NOT operator arms -- they cannot be cited as "ours vs spline"; that cell would
    # be spline compared against itself. They measure two other things:
    #   (a) whether cadence+magscale generalises to a head with no rate conditioning at all, and
    #   (b) don_v2 absolute retention at 40 Hz against its own 20 Hz native.
    # m3_deeponet_s0 is 8300 steps, budget-matched to asrc_s0, so don_v2_native_20env vs
    # asrc_native_20env is a fair head comparison -- and it has never been run.
    "don_v2_cadmag_spline_40env": ("don_v2", 40, 20, "spline+magscale+cadence"),
    "don_v2_zoh_40env":           ("don_v2", 40, 20, "hold"),
    "flow_m1_native_20env": ("flow_m1", 20, None, None),
    # Flow at 40 Hz. It has no operator head, no velocity field and no dt, so it CANNOT fold --
    # cubic-spline resampling of its own 20 Hz chunk is the best it can do. Budget-matched to
    # asrc_s0 at 8300 steps on the same dataset, and it gets the same cadence+magscale fixes.
    "flow_m1_cadmag_spline_40env": ("flow_m1", 40, None, "spline+magscale+cadence"),
    "flow_m1_cadence_40env":       ("flow_m1", 40, None, "magscale+cadence"),
    # --- the 2x2 that is actually runnable: {v2, flow} x {native 20 Hz, spline 40 Hz} ---
    # FOLDING IS NOT IN THIS GRID AND CANNOT BE. Folding integrates a velocity field at a
    # chosen dt; neither v2 nor flow has a velocity field -- both emit a chunk directly.
    # There is no post-hoc folding operator, so "v2+folding" and "flow+folding" are not
    # missing runs, they are undefined. Cubic spline is the best either can do off-native.
    "v2_spatial_native_20env":            ("v2_spatial", 20, 20, None),
    "v2_spatial_cadmag_spline_40env":     ("v2_spatial", 40, 20, "spline+magscale+cadence"),
    "v2_object_native_20env":            ("v2_object", 20, 20, None),
    "v2_object_cadmag_spline_40env":     ("v2_object", 40, 20, "spline+magscale+cadence"),
    "v2_long_native_20env":            ("v2_long", 20, 20, None),
    "v2_long_cadmag_spline_40env":     ("v2_long", 40, 20, "spline+magscale+cadence"),
    "flow30_spatial_native_20env":        ("flow30_spatial", 20, None, None),
    "flow30_spatial_cadmag_spline_40env": ("flow30_spatial", 40, None, "spline+magscale+cadence"),
    "flow30_object_native_20env":        ("flow30_object", 20, None, None),
    "flow30_object_cadmag_spline_40env": ("flow30_object", 40, None, "spline+magscale+cadence"),
    "flow30_long_native_20env":        ("flow30_long", 20, None, None),
    "flow30_long_cadmag_spline_40env": ("flow30_long", 40, None, "spline+magscale+cadence"),
    # Per-suite asrc arms. Run the NATIVE reference FIRST: if it is ~0 the suite is not
    # evaluable for that checkpoint and the rate arms carry no information (that gate already
    # saved ~3 GPU-hours once). Only then run ours-vs-spline.
    "asrc_object_native_20env":         ("asrc_object", 20, 20, None),
    "asrc_object_cadmag_folding_30env": ("asrc_object", 30, 30, "magscale+cadence"),
    "asrc_object_cadmag_spline_30env":  ("asrc_object", 30, 20, "spline+magscale+cadence"),
    "asrc_object_cadmag_folding_40env": ("asrc_object", 40, 40, "magscale+cadence"),
    "asrc_object_cadmag_spline_40env":  ("asrc_object", 40, 20, "spline+magscale+cadence"),
    "asrc_goal_native_20env":         ("asrc_goal", 20, 20, None),
    "asrc_goal_cadmag_folding_40env": ("asrc_goal", 40, 40, "magscale+cadence"),
    "asrc_goal_cadmag_spline_40env":  ("asrc_goal", 40, 20, "spline+magscale+cadence"),
    "asrc_long_native_20env":         ("asrc_long", 20, 20, None),
    "asrc_long_cadmag_folding_40env": ("asrc_long", 40, 40, "magscale+cadence"),
    "asrc_long_cadmag_spline_40env":  ("asrc_long", 40, 20, "spline+magscale+cadence"),
    "flow_native_20env": ("flow", 20, None, None),
    "flow_naive_40env":  ("flow", 40, None, None),
    "flow_spline_40env": ("flow", 40, None, "spline"),
    # proof40: 40Hz-trained head, env 40Hz, NO multirate method (no cadence/magscale/spline)
    "asrc40ft_folding_40env": ("asrc40ft", 40, 40, None),
}


def patch_control_freq(freq: int, horizon: int | None = None) -> None:
    """Force LIBERO's sim to actually run at `freq`.

    LiberoEnv builds OffScreenRenderEnv from a module-global symbol, so replacing that
    symbol injects control_freq without editing site-packages. ControlEnv.__init__ takes
    control_freq=20 as a real parameter; LiberoEnv simply never forwards it.
    """
    import lerobot.envs.libero as libero_module

    original = getattr(libero_module, "_ORIGINAL_OSRE", None)
    if original is None:
        original = libero_module.OffScreenRenderEnv
        libero_module._ORIGINAL_OSRE = original

    def factory(**kwargs):
        kwargs["control_freq"] = freq
        if horizon is not None:
            # only ever RAISE it: robosuite defaults to 1000 and the harness's own MAX_STEPS
            # caps the episode long before this, so existing arms are unaffected.
            kwargs["horizon"] = max(1000, int(horizon))
        return original(**kwargs)

    libero_module.OffScreenRenderEnv = factory


def resample_delta_chunk(chunk: torch.Tensor, target_len: int, native_len: int,
                         offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Cubic-spline resample a (B, T, A) *delta* chunk to target_len steps.

    The pose channels are rescaled by native_len/target_len so total commanded displacement
    is preserved: the action space is relative (controller.use_delta = True), so resampling a
    delta sequence to 2x length without rescaling would double the displacement.

    The rescale MUST happen in raw action space. Normalization is `(raw - offset)/scale`
    applied per step, so multiplying the normalized value by 0.5 does not halve the physical
    delta -- it halves `(raw - offset)/scale`, which is a different quantity. An earlier
    version scaled in normalized space and silently handicapped every spline arm.
    The gripper is a direct command, not a delta, so it is resampled but never scaled.
    """
    if chunk.shape[1] == target_len:
        return chunk
    off = offset[:POSE_DIMS].detach().float().cpu().numpy()
    sc = scale[:POSE_DIMS].detach().float().cpu().numpy()
    source = chunk.float().cpu().numpy()
    raw = source.copy()
    raw[:, :, :POSE_DIMS] = source[:, :, :POSE_DIMS] * sc + off

    x_source = np.linspace(0.0, 1.0, raw.shape[1])
    x_target = np.linspace(0.0, 1.0, target_len)
    out = CubicSpline(x_source, raw, axis=1)(x_target)
    out[:, :, :POSE_DIMS] *= native_len / target_len          # preserve displacement, in RAW space
    out[:, :, :POSE_DIMS] = (out[:, :, :POSE_DIMS] - off) / sc  # back to normalized
    return torch.from_numpy(out).to(device=chunk.device, dtype=chunk.dtype)


# ===========================================================================================
# Harness v2 additions -- see output/k2_campaign_2026-08-28/HARNESS_V2_SPEC.md
# C1 decimate<k>, C2 resamplers ported VERBATIM from the sealed open-loop harness
# ~/k2_campaign_20260828/sweep_v6.py (sha256 93bc4d9642160b35940e8676190e5934e7a0af4fe65b3a28b7a6f0c9438d8b1f).
#
# WHY PORT THE SPLINE TOO: resample_delta_chunk above interpolates the DELTAS and then
# rescales by native_len/target_len. The open-loop baseline interpolates the CUMULATIVE path
# and differences it. Those are different algorithms. Mixing a delta-space spline against a
# cumulative-space TAC-Fold would confound algorithm family with resampler identity, so the
# v2 contrast uses the ported trio only. The legacy `spline` token is left untouched so every
# existing arm reproduces bit-for-bit.
# ===========================================================================================

def _v2_akima_slopes(x, y):
    n, d = y.shape
    if n <= 2:
        dx = x[1] - x[0]
        slope = (y[1:] - y[:-1]) / dx
        return np.repeat(slope, n, axis=0)
    dx = np.diff(x)
    m = np.diff(y, axis=0) / dx[:, None]
    m_pad = np.empty((n + 3, d), dtype=y.dtype)
    m_pad[2:-2] = m
    m_pad[1] = 2.0 * m[0] - m[1]
    m_pad[0] = 2.0 * m_pad[1] - m[0]
    m_pad[-2] = 2.0 * m[-1] - m[-2]
    m_pad[-1] = 2.0 * m_pad[-2] - m[-1]
    dm = np.abs(np.diff(m_pad, axis=0))
    w1, w2 = dm[2:], dm[:-2]
    ws = w1 + w2
    zero = ws < 1e-12
    ws_safe = np.where(zero, 1.0, ws)
    slopes = (w1 * m_pad[1:-2] + w2 * m_pad[2:-1]) / ws_safe
    return np.where(zero, 0.5 * (m_pad[1:-2] + m_pad[2:-1]), slopes)


def _v2_hermite(x0, x1, y0, y1, d0, d1, x):
    h = x1 - x0
    t = (x - x0) / h
    t2 = t * t
    t3 = t2 * t
    h00 = (2.0 * t3 - 3.0 * t2 + 1.0)[:, None]
    h10 = (t3 - 2.0 * t2 + t)[:, None] * h
    h01 = (-2.0 * t3 + 3.0 * t2)[:, None]
    h11 = (t3 - t2)[:, None] * h
    return h00 * y0[None, :] + h10 * d0[None, :] + h01 * y1[None, :] + h11 * d1[None, :]


def _v2_tac_fold(delta, gripper, target_len):
    source_len = len(delta)
    ratio = target_len // source_len
    d = delta.shape[1]
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, d), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0)
    slopes = _v2_akima_slopes(source_t, cumulative)
    out_cum = np.empty((target_len + 1, d), dtype=np.float64)
    out_cum[0] = cumulative[0]
    for i in range(source_len):
        x0, x1 = source_t[i], source_t[i + 1]
        sub_t = np.linspace(x0, x1, ratio + 1)[1:]
        out_cum[i * ratio + 1:(i + 1) * ratio + 1] = _v2_hermite(
            x0, x1, cumulative[i], cumulative[i + 1], slopes[i], slopes[i + 1], sub_t)
    pose = np.diff(out_cum, axis=0)
    if gripper is not None:
        return np.concatenate([pose, np.repeat(gripper, ratio, axis=0)], axis=1)
    return pose


def _v2_cubic_spline(delta, gripper, target_len):
    source_len = len(delta)
    source_t = np.arange(source_len + 1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, delta.shape[1]), dtype=np.float64), np.cumsum(delta, axis=0)], axis=0)
    cs = CubicSpline(source_t, cumulative, axis=0)
    target_cum = cs(np.linspace(0.0, float(source_len), target_len + 1))
    pose = np.diff(target_cum, axis=0)
    if gripper is not None:
        idx = np.minimum(np.floor(np.arange(target_len) * source_len / target_len).astype(int), source_len - 1)
        return np.concatenate([pose, gripper[idx]], axis=1)
    return pose


def _v2_exact_integral(delta, gripper, target_len):
    ratio = target_len // len(delta)
    pose = np.repeat(delta / ratio, ratio, axis=0)
    if gripper is not None:
        return np.concatenate([pose, np.repeat(gripper, ratio, axis=0)], axis=1)
    return pose


_V2_RESAMPLERS = {"tac_fold": _v2_tac_fold, "spline_cum": _v2_cubic_spline, "zoh_fold": _v2_exact_integral}


def v2_transform_chunk(chunk, k, resampler, target_len, offset, scale):
    """C1+C2. RAW-space decimate-by-k then resample to target_len.

    RAW space is mandatory: normalization is (raw - offset)/scale applied PER STEP, so summing
    k normalized deltas does not sum k physical deltas -- the offset accumulates with the step
    count. Same discipline as resample_delta_chunk.

    No native_len/target_len rescale is applied: all three ported resamplers work on the
    cumulative path (or repeat delta/ratio) and are integral-preserving by construction, unlike
    the legacy delta-space spline which needs the correction. Applying it here would silently
    halve the commanded displacement.

    Returns (out_chunk, in_len, dec_len, out_len). The caller gates on dec_len: for the real
    Cell D arms the chunk goes 50 -> 25 -> 50, so in_len == out_len and a naive
    in_len != out_len check would fire on a perfectly live arm. What makes an arm live is that
    the COARSENING happened (dec_len < in_len) and the resampler restored target_len.
    """
    off = offset[:POSE_DIMS].detach().float().cpu().numpy()
    sc = scale[:POSE_DIMS].detach().float().cpu().numpy()
    src = chunk.float().cpu().numpy()
    raw = src.copy()
    raw[:, :, :POSE_DIMS] = src[:, :, :POSE_DIMS] * sc + off
    in_len = raw.shape[1]
    dec_len = in_len
    outs = []
    for b in range(raw.shape[0]):
        pose = raw[b, :, :POSE_DIMS].astype(np.float64)
        rest = raw[b, :, POSE_DIMS:]
        if k > 1:
            nb = pose.shape[0] // k
            pose = pose[:nb * k].reshape(nb, k, POSE_DIMS).sum(axis=1)
            rest = rest[:nb * k].reshape(nb, k, rest.shape[1])[:, 0, :]
        dec_len = pose.shape[0]
        got = resampler(pose, rest.astype(np.float64) if rest.shape[1] else None, target_len)
        outs.append(got)
    out = np.stack(outs, axis=0)
    out[:, :, :POSE_DIMS] = (out[:, :, :POSE_DIMS] - off) / sc
    return (torch.from_numpy(out.astype(np.float32)).to(device=chunk.device, dtype=chunk.dtype),
            in_len, dec_len, out.shape[1])


def build_arm(arm: str, trials_per_task: int, out_dir: Path, results: dict,
              suite: str = "libero_spatial", results_name: str = "multirate_honest.json") -> None:
    model_key, env_freq, head_rate, transform = ARMS[arm]
    backbone, checkpoint, deeponet_head, fourier = MODELS[model_key]
    budget = WALLCLOCK_OVERRIDE or SUITE_WALLCLOCK.get(suite, WALLCLOCK_S)
    steps = int(round(budget * env_freq))
    # A fixed STEP count (not a fixed wall-clock) is what the original protocol used, so the
    # override is in steps -- matching replan=5 exactly rather than approximating it.
    replan = REPLAN_OVERRIDE_STEPS or int(round(REPLAN_S * env_freq))

    patch_control_freq(env_freq, horizon=steps + 200)

    if backbone == "deeponet":
        os.environ["DEEPONET_HEAD"] = deeponet_head
        os.environ["DEEPONET_FOURIER"] = str(fourier)
        # don_v2 (m3_deeponet_s0) has p=64; the ti/til/asrc heads use the 256 default.
        os.environ["DEEPONET_P"] = "64" if model_key == "don_v2" else "256"
        # Eight chronological states belong to the rate-integrated / ncde heads only. The v2
        # head declares state_history_steps=None and its checkpoints carry n_obs_steps=1;
        # forcing 8 there feeds it a state tensor it never saw in training, costs ~54 pp,
        # and passes the tensor guard because no deeponet.* shape changes.
        os.environ["DEEPONET_STATE_HISTORY_STEPS"] = (
            "8" if deeponet_head in {"ti", "til", "asrc", "ncde_style"} else "1")
    else:
        os.environ.pop("DEEPONET_HEAD", None)
        os.environ.pop("DEEPONET_FOURIER", None)
        os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "1"

    # Imported after the env vars are set; load_policy reads them at call time.
    import evaluate_height_screen as screen
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    # load_policy and _rollout read these as module globals, so the rate-dependent
    # budget has to be installed before either is called.
    # Cross-suite eval: only the SUITE changes. DATASET stays libero_spatial because the
    # normalization stats must match TRAINING, not the target suite.
    screen.SUITE = suite
    screen.MAX_STEPS = steps
    screen.REPLAN = replan
    screen.CONTROL_FREQ = head_rate if head_rate is not None else int(NATIVE_HZ)
    screen.DATASET = MODEL_DATASET.get(model_key, DEFAULT_DATASET)
    print(f"[stats] {arm}: normalizing with {screen.DATASET}", flush=True)

    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    if backbone == "deeponet" and not Path(checkpoint).is_dir():
        raise SystemExit(f"[FATAL] {arm}: checkpoint not found: {checkpoint}")
    policy, (preprocessor, postprocessor) = screen.load_policy(backbone, checkpoint, stats)
    # Rate-integrated heads only. DeepONetHeadV2 and the flow backbone have none, so this
    # stays empty for them and every `heads` consumer below must handle that.
    heads = []

    # Action offset/scale for the raw-space rescales below. load_policy configures the
    # DeepONet head from these very stats, so this is identical to reading them off the head
    # -- asserted, not assumed -- and it also works for the flow backbone, which has no head.
    # must live on the policy's device: the deeponet path read these off head buffers that
    # were already on cuda, so CPU tensors here mix devices inside the magscale hook.
    _pol_dev = next(policy.parameters()).device
    A_OFF = torch.as_tensor(stats["action"]["mean"], dtype=torch.float32).flatten().to(_pol_dev)
    A_SC = torch.as_tensor(stats["action"]["std"], dtype=torch.float32).flatten().to(_pol_dev)
    if A_OFF.numel() < POSE_DIMS + 1 or not torch.isfinite(A_OFF).all() or not torch.isfinite(A_SC).all():
        raise SystemExit(f"[FATAL] {arm}: bad action stats from {screen.DATASET}")

    if policy.config.n_action_steps != replan:
        raise SystemExit(f"[FATAL] {arm}: n_action_steps={policy.config.n_action_steps} != {replan}")

    # Verify the head really re-integrates at head_rate. load_policy already called
    # set_rate(CONTROL_FREQ); this asserts it rather than trusting a silent guard.
    if backbone == "deeponet" and deeponet_head in {"ti", "til", "asrc"}:
        from rate_integrated_deeponet import RateIntegratedDeepONetHead
        heads = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)]
        if len(heads) != 1:
            raise SystemExit(f"[FATAL] {arm}: expected 1 rate-integrated head, found {len(heads)}")
        if float(heads[0].runtime_rate_hz) != float(head_rate):
            raise SystemExit(f"[FATAL] {arm}: head rate {heads[0].runtime_rate_hz} != {head_rate}")
        if heads[0].variant != deeponet_head:
            raise SystemExit(f"[FATAL] {arm}: head variant {heads[0].variant} != {deeponet_head}")
        # til forces n_fourier to 0 regardless of the flag; asrc keeps it. Either way the
        # built head must match what the checkpoint was trained with.
        expected_fourier = fourier if deeponet_head == "asrc" else 0
        if heads[0].n_fourier != expected_fourier:
            raise SystemExit(f"[FATAL] {arm}: n_fourier={heads[0].n_fourier} != {expected_fourier}")
        print(f"[{arm}] variant={heads[0].variant} n_fourier={heads[0].n_fourier} "
              f"re-integrates at {heads[0].runtime_rate_hz} Hz "
              f"(native ODE re-integration -- this is NOT TAC-Fold and NOT alias folding; "
              f"TAC-Fold is the 'tac_fold' v2 token)")

    steps_ = transform.split("+") if transform else []

    if "anchor" in steps_:
        if backbone != "deeponet":
            raise SystemExit(f"[FATAL] {arm}: anchor needs the deeponet head")
        if not heads:
            raise SystemExit(f"[FATAL] {arm}: anchor needs a RateIntegratedDeepONetHead; "
                             f"this checkpoint has none, the hook would be a no-op")
        anc = {"n": 0}
        for _h in heads:
            def anchored(c, pose, time_s, rate_hz, _f=_h._feature):
                anc["n"] += 1
                return _f(c, pose, time_s, NATIVE_HZ)
            _h._feature = anchored
        policy._anchor_counter = anc
        print(f"[{arm}] anchor: dt=1/{env_freq} integration, trunk rate token pinned to "
              f"{NATIVE_HZ:.0f} Hz (in-distribution field query, zero interpolation)")

    if "spline" in steps_:
        target_len = int(np.ceil(HORIZON_S * env_freq))  # match what the head would emit natively
        s_off, s_sc = A_OFF.clone(), A_SC.clone()
        # Only RateIntegratedDeepONetHead stores action stats and uses them inside RK4, so
        # only there can a dataset/head mismatch change the dynamics. v2 and flow normalize
        # via the pre/postprocessor, so there is nothing to cross-check.
        if backbone == "deeponet" and heads:   # must equal the head's own stats
            if not torch.allclose(s_off.cpu(), heads[0].action_offset[:s_off.numel()].detach().float().cpu(), atol=1e-6) \
               or not torch.allclose(s_sc.cpu(), heads[0].action_scale[:s_sc.numel()].detach().float().cpu(), atol=1e-6):
                raise SystemExit(f"[FATAL] {arm}: dataset stats != head stats; refusing to rescale")
        raw_get_chunk = policy._get_action_chunk
        applied = {"n": 0}
        if any("eart" in s for s in steps_):
            # EART needs the ORIGINAL (native-rate) gripper stream to time events.
            policy._eart_native_getter = raw_get_chunk

        def spline_chunk(*args, **kwargs):
            chunk = raw_get_chunk(*args, **kwargs)
            out = resample_delta_chunk(chunk, target_len, chunk.shape[1], s_off, s_sc)
            applied["n"] += 1
            return out

        policy._get_action_chunk = spline_chunk
        policy._spline_counter = applied
        print(f"[{arm}] spline resample -> {target_len} steps, pose deltas rescaled")

    _v2_dec = [t for t in steps_ if t.startswith("decimate")]
    _v2_res = [t for t in steps_ if t in _V2_RESAMPLERS]
    if _v2_dec or _v2_res:
        if len(_v2_res) != 1:
            raise SystemExit(f"[FATAL] {arm}: v2 needs exactly one resampler token "
                             f"from {sorted(_V2_RESAMPLERS)}, got {_v2_res}")
        if "spline" in steps_:
            raise SystemExit(f"[FATAL] {arm}: legacy 'spline' cannot combine with v2 tokens")
        _v2_k = int(_v2_dec[0][len("decimate"):]) if _v2_dec else 1
        _v2_fn = _V2_RESAMPLERS[_v2_res[0]]
        _v2_target = int(np.ceil(HORIZON_S * env_freq))
        _v2_off, _v2_sc = A_OFF.clone(), A_SC.clone()
        if backbone == "deeponet" and heads:
            if not torch.allclose(_v2_off.cpu(), heads[0].action_offset[:_v2_off.numel()].detach().float().cpu(), atol=1e-6) \
               or not torch.allclose(_v2_sc.cpu(), heads[0].action_scale[:_v2_sc.numel()].detach().float().cpu(), atol=1e-6):
                raise SystemExit(f"[FATAL] {arm}: dataset stats != head stats; refusing to rescale")
        _v2_prev = policy._get_action_chunk
        _v2_stat = {"n": 0, "coarsened": 0, "in_len": None, "dec_len": None, "out_len": None,
                    "k": _v2_k, "resampler": _v2_res[0]}

        def _v2_chunk(*args, **kwargs):
            c = _v2_prev(*args, **kwargs)
            out, il, dl, ol = v2_transform_chunk(c, _v2_k, _v2_fn, _v2_target, _v2_off, _v2_sc)
            _v2_stat["n"] += 1
            _v2_stat["in_len"], _v2_stat["dec_len"], _v2_stat["out_len"] = il, dl, ol
            if dl < il and ol == _v2_target:
                _v2_stat["coarsened"] += 1
            return out

        policy._get_action_chunk = _v2_chunk
        policy._v2_counter = _v2_stat
        print(f"[{arm}] v2: decimate k={_v2_k} -> {_v2_res[0]} -> {_v2_target} steps "
              f"(RAW space, integral-preserving, no delta rescale)")

    if "magscale" in steps_:
        # Restore the NATIVE per-step delta magnitude. Measured fact: for a fixed commanded
        # delta, LIBERO's OSC delivers roughly the same displacement per SECOND at 20 and
        # 40 Hz, so per-step displacement is ~halved at 40 Hz. A rate-invariant integral
        # (half the delta, twice the steps) therefore under-travels ~2x. Scaling pose deltas
        # by env_freq/NATIVE_HZ restores the travel rate.
        # Must be done in RAW space: normalization subtracts offset per step, so scaling the
        # normalized value is not scaling the physical delta.
        # MAGSCALE_K env override added 2026-08-21 for the 10 Hz calibration sweep
        # (OSC displacement grows as rate falls, so k = f/20 may over-correct).
        k = float(os.environ.get("MAGSCALE_K", env_freq / NATIVE_HZ))
        off = A_OFF[:POSE_DIMS].clone()
        sc = A_SC[:POSE_DIMS].clone()
        # Same reasoning as the spline block: only a rate-integrated head carries these
        # stats internally. v2/flow normalize via the pre/postprocessor, nothing to check.
        if backbone == "deeponet" and heads:
            if not torch.allclose(off.cpu(), heads[0].action_offset[:POSE_DIMS].detach().float().cpu(), atol=1e-6):
                raise SystemExit(f"[FATAL] {arm}: dataset stats != head stats; refusing to rescale")
        prev_chunk = policy._get_action_chunk
        mag = {"n": 0}

        def mag_chunk(*args, **kwargs):
            chunk = prev_chunk(*args, **kwargs)
            out = chunk.clone()
            _sc = sc.to(device=chunk.device, dtype=chunk.dtype)
            _off = off.to(device=chunk.device, dtype=chunk.dtype)
            raw = chunk[..., :POSE_DIMS] * _sc + _off
            out[..., :POSE_DIMS] = (raw * k - _off) / _sc
            mag["n"] += 1
            return out

        policy._get_action_chunk = mag_chunk
        policy._mag_counter = mag
        print(f"[{arm}] magscale: raw pose deltas x{k:g} (restores native per-step travel)")

    if "eart" in steps_:
        # EART (Event-Aligned Rate Retargeting) 2026-08-21:
        # every existing resampler (cubic spline, TAC-Fold, freq-domain) treats the
        # gripper channel as a continuous signal and smears/delays its open-close
        # EVENT by up to a full cell (0-100 ms at 10 Hz). EART re-times the gripper
        # to the majority-of-time state of each target cell: the command reflects the
        # state at the cell MIDPOINT, so every crossing lands within +/- half a cell
        # (<= 50 ms at 10 Hz) and never a full cell late.
        native_getter = getattr(policy, "_eart_native_getter", None)
        if native_getter is None:
            raise SystemExit("[FATAL] eart requires the spline transform to run first")
        # NOTE: must NOT reuse the name `prev_chunk` here -- magscale's closure captures
        # that name by reference and would start calling itself (recursion bug 2026-08-21).
        pose_chunk_fn = policy._get_action_chunk
        state = {"n": 0}

        def eart_chunk(*args, **kwargs):
            native = native_getter(*args, **kwargs)       # native-rate chunk (B, 50, 7)
            out = pose_chunk_fn(*args, **kwargs)          # pose: spline+magscale'd
            src = native[..., POSE_DIMS].detach().float().cpu().numpy()  # (B, T) gripper
            B, T = src.shape
            target_len = out.shape[1]
            ratio = T / float(target_len)
            idx = np.clip(np.floor((np.arange(target_len) + 0.5) * ratio).astype(int), 0, T - 1)
            g_new = torch.as_tensor(src[:, idx], device=out.device, dtype=out.dtype)
            out = torch.cat([out[..., :POSE_DIMS], g_new.unsqueeze(-1)], dim=-1)
            state["n"] += 1
            return out

        policy._get_action_chunk = eart_chunk
        policy._eart_counter = state
        print(f"[{arm}] EART: gripper re-timed to cell midpoints "
              f"(event error <= {0.5 / env_freq * 1000:.0f} ms)")

    if "eart2" in steps_:
        # EART v2 (2026-08-21): TRUE majority-of-time rule with crossing interpolation.
        # v1 sampled the cell-END native value (events could fire a full cell EARLY --
        # killed task 5). v2 detects the 0-crossing between the native steps a cell spans
        # and picks the state that holds for the majority of the cell's duration.
        native_getter2 = getattr(policy, "_eart_native_getter", None)
        if native_getter2 is None:
            raise SystemExit("[FATAL] eart2 requires the spline transform to run first")
        pose_chunk_fn2 = policy._get_action_chunk
        state2 = {"n": 0}

        def eart2_chunk(*args, **kwargs):
            native = native_getter2(*args, **kwargs)
            out = pose_chunk_fn2(*args, **kwargs)
            src = native[..., POSE_DIMS].detach().float().cpu().numpy()  # (B, T)
            B, T = src.shape
            target_len = out.shape[1]
            ratio = T / float(target_len)
            g_out = np.empty((B, target_len), dtype=src.dtype)
            for k in range(target_len):
                lo = k * ratio
                hi = (k + 1) * ratio
                n0 = int(np.floor(lo))
                n1 = int(np.floor(hi))
                if n0 == n1:
                    g_out[:, k] = src[:, n0]
                    continue
                g0 = src[:, n0]
                g1 = src[:, min(n0 + 1, T - 1)]
                denom = g1 - g0
                safe = np.abs(denom) > 1e-9
                cross_frac = np.where(safe, (0.0 - g0) / np.where(safe, denom, 1.0), 1.5)
                t_e = n0 + np.clip(cross_frac, 0.0, 1.0)
                mid = (lo + hi) / 2.0
                use_post = t_e < mid
                g_out[:, k] = np.where(use_post, g1, g0)
            g_new = torch.as_tensor(g_out, device=out.device, dtype=out.dtype)
            out = torch.cat([out[..., :POSE_DIMS], g_new.unsqueeze(-1)], dim=-1)
            state2["n"] += 1
            return out

        policy._get_action_chunk = eart2_chunk
        policy._eart2_counter = state2
        print(f"[{arm}] EART-v2: majority-of-time gripper with crossing interpolation")

    if "hold" in steps_ or "cadence" in steps_:
        # ponytail: wrapping select_action beats writing a second rollout loop. Skipping the
        # call on off-cycle steps also leaves the observation queue unfed, which is exactly what
        # restores the 0.4 s history span -- one wrapper fixes cadence and travel rate both.
        from lerobot.utils.constants import ACTION

        # Phase accumulator, not a modulo: env_freq/NATIVE_HZ need not be an integer. At 30 Hz
        # the ratio is 1.5, so the query pattern is 1,2,1,2,... env steps -- exactly 20 Hz on
        # average. int(round(1.5)) == 2 would silently run a 15 Hz observation cadence instead.
        # This is also why integer zero-order hold cannot express a 30 Hz target.
        resolve = "cadence" in steps_
        raw_select = policy.select_action
        state = {"n": 0, "last": None, "queries": 0, "resolved": 0, "starved": 0, "phase": -1}

        def cadence_select(batch):
            phase = int(state["n"] * NATIVE_HZ / env_freq)
            if phase != state["phase"]:
                state["phase"] = phase
                state["last"] = raw_select(batch)
                state["queries"] += 1
            elif resolve and len(policy._queues[ACTION]) > 0:
                # Off-cycle step: take the NEXT planned action instead of repeating a stale one.
                # Popping the queue directly bypasses populate_queues, so the observation cadence
                # stays at NATIVE_HZ -- identical to the hold arm. The only difference from hold
                # is that this intermediate action is resolved rather than held.
                state["last"] = policy._queues[ACTION].popleft()
                state["resolved"] += 1
            elif resolve:
                state["starved"] += 1  # queue empty off-cycle; fall back to holding
            state["n"] += 1
            return state["last"]

        policy.select_action = cadence_select
        policy._hold_state = state
        ratio = env_freq / NATIVE_HZ
        print(f"[{arm}] {'resolved-cadence' if resolve else 'zero-order hold'}: "
              f"{NATIVE_HZ:.0f} Hz decisions in a {env_freq} Hz sim "
              f"(1 query per {ratio:g} env steps"
              f"{', NON-INTEGER: integer hold cannot express this' if ratio % 1 else ''})"
              f"{'; intermediates taken from the plan' if resolve else ''}")

    arm_config = {"env_freq": env_freq, "head_rate": head_rate, "transform": transform,
                  "steps": steps, "replan": replan, "wallclock_s": budget,
                  "settle_steps": int(round(SETTLE_STEPS_20HZ * env_freq / NATIVE_HZ)),
                  "suite": suite, "trained_on": "libero_spatial",
                  "model": model_key, "checkpoint": checkpoint,
                  "deeponet_head": deeponet_head, "fourier": fourier}
    entry = results.setdefault(arm, {"config": arm_config, "per_task": {}, "aggregate": None})
    if entry.get("config") != arm_config:
        raise SystemExit(f"[FATAL] {arm}: stored config {entry.get('config')} != {arm_config}; "
                         "refusing to mix protocols -- delete the stale entry")

    results_path = out_dir / results_name
    for task_id in range(N_TASKS):
        task_entry = entry["per_task"].setdefault(str(task_id), {"episodes": []})
        env = screen._make_env(task_id)

        sim_freq = env._env.env.control_freq
        if float(sim_freq) != float(env_freq):
            raise SystemExit(f"[FATAL] {arm}: sim control_freq={sim_freq} != {env_freq}")

        # LiberoEnv.reset settles the scene with num_steps_wait *control* steps, so at
        # 40 Hz the default 10 covers half the physical time it does at 20 Hz. Unscaled,
        # the faster arms would start from less-settled object poses -- a confound that
        # looks exactly like a policy effect. Scale it to hold settle duration fixed.
        env.num_steps_wait = int(round(env.num_steps_wait * env_freq / NATIVE_HZ))
        if task_id == 0:
            print(f"[{arm}] settle window scaled to {env.num_steps_wait} steps "
                  f"({env.num_steps_wait / env_freq:.2f} s at {env_freq} Hz)")

        task_entry["task"] = env.task_description
        for trial in range(len(task_entry["episodes"]), trials_per_task):
            env.init_state_id = trial  # pinned; LiberoEnv.reset advances it by _reset_stride
            assert env.init_state_id == trial
            episode = screen._rollout(policy, preprocessor, postprocessor, env,
                                      env.task_description, 1000 + trial)
            task_entry["episodes"].append(episode)
            task_entry["success_rate"] = float(np.mean([e["success"] for e in task_entry["episodes"]]))
            results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
            print(f"[{arm}] task {task_id} trial {trial:02d}: "
                  f"{'OK' if episode['success'] else 'x'} ({episode['steps']}/{steps} steps)",
                  flush=True)
        env.close()

    if "cadence" in steps_ or "hold" in steps_:
        st = policy._hold_state
        print(f"[{arm}] cadence check: {st['queries']} policy queries, {st['resolved']} resolved "
              f"intermediates, {st['starved']} starved of {st['n']} env steps")
        if "cadence" in steps_ and st["resolved"] == 0:
            raise SystemExit(f"[FATAL] {arm}: no intermediate was ever resolved from the plan; "
                             "this arm is identical to zero-order hold")

    if "spline" in steps_ and policy._spline_counter["n"] == 0:
        raise SystemExit(f"[FATAL] {arm}: spline transform never fired")

    if hasattr(policy, "_v2_counter"):
        _s = policy._v2_counter
        print(f"[{arm}] v2 inertness check: {_s['n']} calls, {_s['coarsened']} coarsened "
              f"({_s['in_len']} -> {_s['dec_len']} -> {_s['out_len']}), k={_s['k']}, "
              f"resampler={_s['resampler']}")
        if _s["n"] == 0:
            raise SystemExit(f"[FATAL] {arm}: v2 transform never fired")
        if _s["coarsened"] == 0:
            raise SystemExit(f"[FATAL] {arm}: v2 transform never coarsened the chunk "
                             f"({_s['in_len']} -> {_s['dec_len']} -> {_s['out_len']}); "
                             "this arm is INERT, not a tie")

    if "anchor" in steps_ and policy._anchor_counter["n"] == 0:
        raise SystemExit(f"[FATAL] {arm}: anchor never fired; the head was queried off-rate")

    eps = [e for t in entry["per_task"].values() for e in t["episodes"]]
    fails = [e["steps"] for e in eps if not e["success"]]
    oks = [e["steps"] for e in eps if e["success"]]
    entry["diagnostics"] = {
        "n": len(eps), "n_success": len(oks),
        "fail_at_cap": sum(1 for x in fails if x >= steps),
        "fail_early": sum(1 for x in fails if x < steps),
        "success_steps_median": float(np.median(oks)) if oks else None,
        "success_steps_max": max(oks) if oks else None,
        "cap": steps,
        # Counters only: _hold_state["last"] holds a torch Tensor, which is not JSON
        # serializable and crashed the whole arm after its rollouts had already run.
        "cadence": ({k: v for k, v in policy._hold_state.items() if isinstance(v, int)}
                    if hasattr(policy, "_hold_state") else None),
        "spline_applied": policy._spline_counter["n"] if hasattr(policy, "_spline_counter") else None,
        "magscale_applied": policy._mag_counter["n"] if hasattr(policy, "_mag_counter") else None,
        "anchor_applied": policy._anchor_counter["n"] if hasattr(policy, "_anchor_counter") else None,
        "v2_transform": (dict(policy._v2_counter) if hasattr(policy, "_v2_counter") else None),
    }
    print(f"[{arm}] diagnostics: {entry['diagnostics']['n_success']}/{len(eps)} ok | "
          f"fail_at_cap={entry['diagnostics']['fail_at_cap']} "
          f"fail_early={entry['diagnostics']['fail_early']} | "
          f"succ_median={entry['diagnostics']['success_steps_median']}/{steps}", flush=True)
    entry["aggregate"] = float(np.mean([t["success_rate"] for t in entry["per_task"].values()]))
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"--> {arm}: {entry['aggregate'] * 100:.1f}% "
          f"({steps} steps @ {env_freq} Hz = {WALLCLOCK_S:.1f} s, replan every {replan})\n")

    del policy
    torch.cuda.empty_cache()


def smoke() -> None:
    """Verification gate: prove the env rate, head rate, and spline all take effect."""
    import evaluate_height_screen as screen
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    for env_freq in (20, 40):
        patch_control_freq(env_freq)
        env = screen._make_env(0)
        got = env._env.env.control_freq
        assert float(got) == float(env_freq), f"control_freq {got} != {env_freq}"
        print(f"[smoke] env control_freq={got} OK ({env_freq} Hz requested)")
        env.close()

    os.environ["DEEPONET_HEAD"] = "til"
    os.environ["DEEPONET_STATE_HISTORY_STEPS"] = "8"
    screen.REPLAN = 20
    screen.MAX_STEPS = 440
    screen.CONTROL_FREQ = 40
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preprocessor, postprocessor) = screen.load_policy("deeponet", DEEPONET_CKPT, stats)

    from rate_integrated_deeponet import RateIntegratedDeepONetHead
    head = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)][0]
    assert float(head.runtime_rate_hz) == 40.0, head.runtime_rate_hz
    assert head.n_fourier == 0, head.n_fourier
    print(f"[smoke] head rate={head.runtime_rate_hz} n_fourier={head.n_fourier} OK")

    # resample_delta_chunk is pure, so check its contract on a synthetic delta chunk:
    # summed pose displacement must be preserved, and the gripper must NOT be rescaled.
    synthetic = torch.ones(1, 50, 7)
    doubled = resample_delta_chunk(synthetic, 100, 50, torch.zeros(POSE_DIMS), torch.ones(POSE_DIMS))
    assert doubled.shape == (1, 100, 7), doubled.shape
    pose_before = synthetic[0, :, :POSE_DIMS].sum(0)
    pose_after = doubled[0, :, :POSE_DIMS].sum(0)
    assert torch.allclose(pose_before, pose_after, atol=1e-4), (pose_before, pose_after)
    assert torch.allclose(doubled[0, :, POSE_DIMS], torch.ones(100), atol=1e-4)
    print(f"[smoke] spline 50->100 preserves pose displacement "
          f"({pose_before[0]:.3f} -> {pose_after[0]:.3f}) and leaves gripper unscaled")

    # Drive the real select_action path so the 8-step state history fills, and record the
    # chunk from inside it -- this is exactly where the spline arm hooks in.
    patch_control_freq(40)
    env = screen._make_env(0)
    env.init_state_id = 0
    obs, _ = env.reset(seed=1000)
    seen = []
    raw_get_chunk = policy._get_action_chunk
    policy._get_action_chunk = lambda *a, **k: (lambda c: (seen.append(tuple(c.shape)), c)[1])(
        raw_get_chunk(*a, **k))

    for _ in range(screen.REPLAN + 2):
        batch = preprocessor(screen._policy_input(obs, env.task_description))
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch)
        obs, _, _, _, _ = env.step(postprocessor(action).to("cpu").float().numpy().reshape(-1))

    assert seen, "no chunk was ever produced"
    print(f"[smoke] chunks produced inside select_action: {seen}")
    assert seen[0][1] == 100, f"expected T=100 at 40 Hz (2.5 s horizon), got {seen[0]}"
    assert len(seen) >= 2, f"replan never fired in {screen.REPLAN + 2} steps: {seen}"
    print(f"[smoke] {screen.REPLAN + 2} env steps at 40 Hz, replanned {len(seen)}x OK")
    env.close()
    print("[smoke] ALL CHECKS PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials_per_task", type=int, default=15)
    parser.add_argument("--arms", default="all", help="comma-separated arm names, or 'all'")
    parser.add_argument("--out", default="multirate_honest_out")
    parser.add_argument("--wallclock", type=float, default=None,
                        help="override the per-suite episode budget in seconds")
    parser.add_argument("--suite", default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        help="checkpoints trained on libero_spatial; others are cross-suite OOD")
    parser.add_argument("--smoke", action="store_true", help="run verification gate only")
    parser.add_argument("--replan_steps", type=int, default=None,
                        help="fixed replan interval in env steps; overrides REPLAN_S")
    args = parser.parse_args()
    if args.wallclock is not None:
        global WALLCLOCK_OVERRIDE
        WALLCLOCK_OVERRIDE = args.wallclock

    if args.smoke:
        smoke()
        return

    selected = list(ARMS) if args.arms == "all" else args.arms.split(",")
    unknown = [a for a in selected if a not in ARMS]
    if unknown:
        raise SystemExit(f"[FATAL] unknown arms: {unknown}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the historical filename for libero_spatial so completed work is not redone.
    results_name = ("multirate_honest.json" if args.suite == "libero_spatial"
                    else f"multirate_{args.suite}.json")
    results_path = out_dir / results_name
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    if args.replan_steps is not None:
        if args.replan_steps < 1:
            raise SystemExit("[FATAL] --replan_steps must be >= 1")
        global REPLAN_OVERRIDE_STEPS
        REPLAN_OVERRIDE_STEPS = int(args.replan_steps)
        print(f"[replan] OVERRIDE: {REPLAN_OVERRIDE_STEPS} env steps "
              f"(default would be {int(round(REPLAN_S * 20))} at 20 Hz)", flush=True)
    print(f"[suite] {args.suite} -> {results_path}", flush=True)

    for arm in selected:
        entry = results.get(arm, {})
        # Skip only if the arm is complete AT THIS trial count. Guarding on `aggregate is
        # not None` alone is what froze the old matrix's n=20 arms while others reached
        # n=150, silently producing a leaderboard that mixed sample sizes.
        episodes = [len(t["episodes"]) for t in entry.get("per_task", {}).values()]
        if (entry.get("aggregate") is not None and len(episodes) == N_TASKS
                and min(episodes) >= args.trials_per_task):
            print(f"Skipping {arm} (complete at n={sum(episodes)}: "
                  f"{entry['aggregate'] * 100:.1f}%)")
            continue
        if entry.get("aggregate") is not None:
            print(f"Extending {arm} from n={sum(episodes)} to "
                  f"{N_TASKS * args.trials_per_task}")
            entry["aggregate"] = None
        print(f"\n{'=' * 60}\nARM {arm}\n{'=' * 60}")
        build_arm(arm, args.trials_per_task, out_dir, results, args.suite, results_name)

    print("\nSUMMARY (matched wall-clock, chunk-driven replan)")
    for arm in selected:
        aggregate = results.get(arm, {}).get("aggregate")
        print(f"  {arm:22s} {'n/a' if aggregate is None else f'{aggregate * 100:5.1f}%'}")


if __name__ == "__main__":
    main()
