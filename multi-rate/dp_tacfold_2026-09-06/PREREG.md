# Pre-registration — closed-loop Diffusion Policy + resamplers (written 2026-09-06 ~03:55 KST, BEFORE any full run)

Policy: dp_min.DiffusionPolicy (lerobot U-Net vendored, state-only, n_obs=2, horizon=16, exec 8, EMA, DDIM-10), 30k steps, bs 256.
Tasks: ManiSkill PushT-v1, PickCube-v1 (RL demos, pre-clip actions saturate); RoboMimic lift, can, square (human demos, in [-1,1]).
Eval: k=2, n=100, paired (same init per episode across arms; DP noise re-seeded per (episode, replan)).
Arms: native, zoh, spline, spline_satfix, pchip, pchip_satfix, tac_fold, tac_fold_satfix. Stats: exact McNemar vs zoh and vs native.

## Predictions (mechanistic, committed now)
1. RoboMimic: policy raw |a|>1 fraction ≈ 0 → every *_satfix arm ≡ its base arm; all resamplers within noise of native; no contrast survives Holm.
2. ManiSkill (raw sat ≈ 0.6 in smoke): satfix arms beat their un-fixed base arms (satfix has something to fix); tac_fold_satfix ≥ zoh.
   Effect size vs open-loop replay expected SMALLER (replanning every 8 steps caps drift). Sign vs native: uncertain.
3. Interpolant ordering among {spline, pchip, tac_fold} (un-fixed) will NOT be significant anywhere (MSE-equivalent per recon diagnostic).

## What is NOT allowed after seeing results
No re-tuning of steps/arms/k/n; no dropping tasks; no swapping the reference arm. A failed prediction is reported as such.
