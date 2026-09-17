| Diff. configuration | PushT | Lift | PickCube | Can | Square |
|---|---|---|---|---|---|
| Diff. 1X Base | 31±3% | 100% | 90±0% | 98% | 89% |
| Diff. 1X +B-spline head (trained, their method) | 28±7% (-3) | 100% (+0) | 90±1% (-1) | 99% (+1) | 94% (+5) |
| Diff. 1X +B-spline post-hoc (eps .05) | 34% (+8) | 100% (+0) | 89% (-1) | 98% (+0) | 91% (+2) |
| Diff. 1X +QP-eps .05 (ours, post-hoc) | 31% (+5) | 100% (+0) | 90% (+0) | 98% (+0) | 90% (+1) |

| FM configuration | PushT | Lift | PickCube | Can | Square |
|---|---|---|---|---|---|
| FM 1X Base | 30% | 100% | 89% | 100% | 84% |
| FM 1X +B-spline head (trained, their method) | — | — | — | — | — |
| FM 1X +B-spline post-hoc (eps .05) | 35% (+5) | 100% (+0) | 89% (+0) | 100% (+0) | 84% (+0) |
| FM 1X +QP-eps .05 (ours, post-hoc) | 23% (-7) | 100% (+0) | 88% (-1) | 97% (-3) | 86% (+2) |

delta vs same-seed 1X Base, paired exact McNemar (Holm over seeds), * p<0.05. ±sd over 3 seeds where shown; n=100 per seed.
