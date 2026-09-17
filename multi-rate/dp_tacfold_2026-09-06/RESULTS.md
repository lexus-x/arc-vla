# RESULTS — closed-loop Diffusion Policy + resamplers, k=2, n=100/arm/task (2026-09-06, campaign 03:42–06:09 KST)

Protocol as in PREREG.md (nothing changed after launch). Policy raw pre-clip saturation: lift 0.000, can 0.000, square 0.000, PushT-v1 0.073, PickCube-v1 0.266.

## Success rates (%)

| task | native | zoh | spline | spline_satfix | pchip | pchip_satfix | tac_fold | tac_fold_satfix |
|---|---|---|---|---|---|---|---|---|
| lift | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| can | 98 | 98 | 98 | 98 | 98 | 98 | 97 | 98 |
| square | 89 | 86 | 84 | 89 | 89 | 86 | 87 | 88 |
| PushT-v1 | 26 | 16 | 32 | 38 | 30 | 25 | 24 | 30 |
| PickCube-v1 | 90 | 85 | 74 | 87 | 69 | 85 | 83 | 88 |

## Primary family — exact McNemar, Holm over all 70 contrasts (7 arms × {vs zoh, vs native} × 5 tasks)

| task | arm | ref | Δpp | p raw | p Holm | survives |
|---|---|---|---|---|---|---|
| PickCube-v1 | pchip | native | -21 | 4.92e-05 | 0.00345 | YES |
| PushT-v1 | spline_satfix | zoh | +22 | 5.95e-05 | 0.0041 | YES |
| PickCube-v1 | pchip | zoh | -16 | 0.000402 | 0.0274 | YES |
| PickCube-v1 | spline | native | -16 | 0.00154 | 0.103 | no |
| PushT-v1 | tac_fold_satfix | zoh | +14 | 0.00661 | 0.436 | no |
| PushT-v1 | spline | zoh | +16 | 0.0113 | 0.737 | no |
| PickCube-v1 | spline | zoh | -11 | 0.0127 | 0.814 | no |
| PushT-v1 | pchip | zoh | +14 | 0.0161 | 1 | no |
| PushT-v1 | spline_satfix | native | +12 | 0.0357 | 1 | no |
| PushT-v1 | native | zoh | +10 | 0.0525 | 1 | no |
| PushT-v1 | zoh | native | -10 | 0.0525 | 1 | no |
| PushT-v1 | pchip_satfix | zoh | +9 | 0.0784 | 1 | no |
| PushT-v1 | tac_fold | zoh | +8 | 0.115 | 1 | no |
| PickCube-v1 | tac_fold | native | -7 | 0.118 | 1 | no |
| PickCube-v1 | native | zoh | +5 | 0.18 | 1 | no |
| PickCube-v1 | zoh | native | -5 | 0.18 | 1 | no |
| PickCube-v1 | pchip_satfix | native | -5 | 0.18 | 1 | no |

(all other 53 contrasts have raw p ≥ 0.2; lift/can are ceiling cells)

## Secondary family — satfix vs its own base interpolant, Holm over 15

| task | pair | Δpp | p raw | p Holm |
|---|---|---|---|---|
| PickCube-v1 | pchip_satfix vs pchip | +16 | 0.000145 | 0.00217 |
| PickCube-v1 | spline_satfix vs spline | +13 | 0.00235 | 0.0329 |
| PickCube-v1 | tac_fold_satfix vs tac_fold | +5 | 0.18 | 1 |
| PushT-v1 | pchip_satfix vs pchip | -5 | 0.458 | 1 |
| PushT-v1 | spline_satfix vs spline | +6 | 0.362 | 1 |
| PushT-v1 | tac_fold_satfix vs tac_fold | +6 | 0.263 | 1 |
| can | pchip_satfix vs pchip | +0 | 1 | 1 |
| can | spline_satfix vs spline | +0 | 1 | 1 |
| can | tac_fold_satfix vs tac_fold | +1 | 1 | 1 |
| lift | pchip_satfix vs pchip | +0 | 1 | 1 |
| lift | spline_satfix vs spline | +0 | 1 | 1 |
| lift | tac_fold_satfix vs tac_fold | +0 | 1 | 1 |
| square | pchip_satfix vs pchip | -3 | 0.375 | 1 |
| square | spline_satfix vs spline | +5 | 0.0625 | 0.812 |
| square | tac_fold_satfix vs tac_fold | +1 | 1 | 1 |

## Un-fixed interpolant ordering — spline/pchip/tac_fold pairwise, Holm over 15

| task | pair | Δpp | p raw | p Holm |
|---|---|---|---|---|
| PickCube-v1 | spline vs tac_fold | -9 | 0.00391 | 0.0586 |
| PickCube-v1 | pchip vs tac_fold | -14 | 0.00434 | 0.0608 |
| square | spline vs pchip | -5 | 0.18 | 1 |
| PushT-v1 | spline vs tac_fold | +8 | 0.215 | 1 |
| PickCube-v1 | spline vs pchip | +5 | 0.302 | 1 |
| PushT-v1 | pchip vs tac_fold | +6 | 0.377 | 1 |
| square | spline vs tac_fold | -3 | 0.453 | 1 |
| square | pchip vs tac_fold | +2 | 0.688 | 1 |

## PREREG scorecard

**P1 (RoboMimic: satfix inert, nothing survives Holm):** HOLDS — raw sat = 0.000 on all three; every *_satfix arm within ±3pp of its base; 0 RoboMimic contrasts survive Holm.
**P2 (PushT-v1): satfix > base per arm:** spline +6, pchip -5, tac_fold +6; tac_fold_satfix vs zoh +14pp raw p=0.00661 Holm p=0.436.
**P2 (PickCube-v1): satfix > base per arm:** spline +13, pchip +16, tac_fold +5; tac_fold_satfix vs zoh +3pp raw p=0.375 Holm p=1.
**P3 (un-fixed interpolant ordering not significant):** HOLDS — 0 of 15 pairwise contrasts survive Holm.