# Pre-registration -- full grid: {RoboMimic, RoboCasa, PushT/PickCube} x {Diffusion Policy, Flow Matching}
Written 2026-09-07 ~03:00 KST, BEFORE any new (RoboCasa or FM) result is inspected.

## Context
This directly extends `dp_tacfold_2026-09-06/` (RoboMimic lift/can/square + ManiSkill
PushT-v1/PickCube-v1, Diffusion Policy only, DONE, RESULTS.md written). This campaign adds:
1. `bspline`/`bspline_satfix` arms (ported from `satfix_2026-09-05/eval_stream_robomimic.py`) to
   all tasks, DP and FM alike.
2. FlowMatchingPolicy (`fm_min.py`) -- identical U-Net to DiffusionPolicy, straight-line
   conditional flow matching objective, Euler-10 sampler. Run on the SAME RoboMimic + ManiSkill
   tasks as the DP campaign.
3. RoboCasa (`RC-OpenDrawer`, `RC-PnPCounterToStove` -- the same 2 tasks as the earlier n=10
   exploratory GR00T pass) with BOTH DP and FM, real n=15 held-out eval (up from n=10, still far
   below the n=100 used elsewhere -- RoboCasa's `human_raw` sets have only 55 demos/task total,
   see "Known limitation" below).

## Protocol (unchanged from dp_tacfold_2026-09-06 where applicable)
- k=2, paired (same init per episode across arms), DP/FM sampling noise re-seeded per
  (episode, replan) so arms see identical policy proposals whenever their obs agree.
- Arms: native, zoh, spline, spline_satfix, pchip, pchip_satfix, tac_fold, tac_fold_satfix,
  bspline, bspline_satfix (10 total, up from 8 -- bspline is new this campaign).
- RoboMimic / ManiSkill: n_train=200, n_eval=100, 30k train steps (matches dp_tacfold_2026-09-06
  exactly, so DP results for these 5 tasks are NOT re-run -- fm_min results are new).
- RoboCasa: n_train=39, n_eval=15, 15k train steps (see limitation below for why these differ).
- Stats: exact McNemar vs zoh and vs native, per arm, per task, per policy. Holm correction
  across the FULL new-result family (every non-DP/RoboMimic/ManiSkill contrast produced this
  campaign) -- see "Contrast family" below.

## Known limitation, stated before running
RoboCasa `human_raw` datasets have only 55 demonstrated episodes per task (vs hundreds for
RoboMimic/ManiSkill). n_train=39/n_eval=15 is forced by data availability, not choice. A 64M-
parameter U-Net trained on 40 demos is very likely undertrained relative to the RoboMimic/
ManiSkill arms (200 demos) -- expect RoboCasa native/zoh success rates to be noticeably lower
and noisier than the other two benchmark families, and expect any resampler contrast there to
have wide confidence intervals at n=15. This is a genuine data-availability ceiling, not a bug,
and is reported as such regardless of outcome -- augmenting the dataset or changing n after
seeing results is not allowed under this pre-reg.

## Predictions (mechanistic, committed before running)
P1. bspline arms will land close to spline arms in success rate on every task (both are
    unconstrained smoothing splines fit to the same cumulative trajectory; bspline additionally
    smooths via `s>0`, so expect bspline to *undershoot* peaks slightly more than plain cubic
    spline -- i.e. bspline_satfix has less to fix than spline_satfix, similar to how TAC-Fold's
    own Akima damping left less for satfix to fix in the 2026-09-05 RoboMimic campaign).
P2. TAC-Fold does NOT win outright on any task under either policy. Basis: the 2026-09-06
    reconstruction-MSE diagnostic already showed TAC-Fold is MSE-indistinguishable from
    spline/pchip/bspline on RoboMimic/PushT (it IS a member of that Hermite-spline family with
    Akima slope limiting) -- swapping the policy from DP to FM does not change that mechanism,
    since it operates on the resampler's output, not on how the target action chunk was
    produced. Expect "on par with the best smooth interpolant," not a win.
P3. RoboMimic (lift/can/square) FM results replicate the DP finding: raw pre-clip |a|>1 fraction
    ~0 (human teleop demos, in-range by construction) -> satfix arms are inert, no contrast
    survives Holm, consistent with dp_tacfold_2026-09-06 P1.
P4. PushT-v1/PickCube-v1 FM results show SOME saturation-driven separation between arms (same
    mechanism as DP: policy raw outputs exceed |1| pre-clip on these RL-demo-trained tasks), but
    sign/magnitude uncertain a priori (FM's velocity-field sampling noise structure differs from
    DP's DDIM, so effect size need not match DP's).
P5. RoboCasa: given the n=15/n_train=40 limitation above, predict NO contrast survives Holm on
    either RoboCasa task, for either policy -- underpowered, not necessarily null in the
    underlying effect. If something DOES survive Holm at n=15, treat it with extra skepticism
    (multiple-comparisons risk under small n) and flag for a larger confirmatory run before
    citing it anywhere.

## Contrast family for Holm correction
All NEW contrasts this campaign: FM on {lift, can, square, PushT-v1, PickCube-v1} (5 tasks) +
DP+FM on {RC-OpenDrawer, RC-PnPCounterToStove} (2 tasks x 2 policies) = 5 + 4 = 9 task/policy
cells x 9 non-native-non-zoh arms x 2 refs (zoh, native) = up to 162 contrasts, corrected
together as one family (bspline additions to the ALREADY-COMPLETE DP/RoboMimic/ManiSkill runs
are a separate, smaller family: 5 tasks x 2 bspline arms x 2 refs = 20 contrasts, since adding
one new arm to an existing completed campaign is a distinct question from the 9 new cells).
Report both families' full tables, not just what survives.

## What is NOT allowed after seeing results
No re-tuning of steps/arms/k/n_train/n_eval for RoboCasa after seeing eval numbers, no dropping
tasks, no swapping the reference arm, no augmenting the RoboCasa demo count after seeing a
disappointing result. A failed prediction (P1-P5) is reported as failed, not reframed.

## Correction before launch (2026-09-07 03:03 KST, before any training/eval ran)
RoboCasa's own `dataset_states_to_obs.py` conversion + `make_demo_ids_contiguous` step dropped
one demo per task during processing (55 raw -> 54 converted, both tasks) -- normal for this
pipeline (occasional failed re-simulation of a recorded episode). n_train corrected from 40 to
39 (39+15=54, fits exactly). This is a data-availability fix made before seeing any result, not
a post-hoc adjustment; n_eval stays at 15 as originally planned.
