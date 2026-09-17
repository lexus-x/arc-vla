
| Diff. configuration | PushT | Lift | PickCube | Can | Square |
|---|---|---|---|---|---|
| Diff. 1X Base | 26% | 100% | 90% | 98% | 89% |
| Diff. 1X +B-spline post-hoc eps.005 | 30% (+4) | 100% (+0) | 90% (+0) | 98% (+0) | 88% (-1) |
| Diff. 1X +B-spline post-hoc eps.05 | 34% (+8) | 100% (+0) | 89% (-1) | 98% (+0) | 91% (+2) |
| Diff. 1X +QP-eps.05 (ours) | 31% (+5) | 100% (+0) | 90% (+0) | 98% (+0) | 90% (+1) |
| Diff. 1X +QP-eps.10 (ours) | 20% (-6) | 100% (+0) | 89% (-1) | 99% (+1) | 86% (-3) |
| Diff. 2X naive (ZOH) | 16% (-10) | 100% (+0) | 85% (-5) | 98% (+0) | 86% (-3) |
| Diff. 2X +spline+satfix | 38% (+12*) | 100% (+0) | 87% (-3) | 98% (+0) | 89% (+0) |
| Diff. 2X +TAC-Fold+satfix | 30% (+4) | 100% (+0) | 88% (-2) | 98% (+0) | 88% (-1) |
| Diff. 2X +QP (ours) | 20% (-6) | 100% (+0) | 87% (-3) | 97% (-1) | 90% (+1) |
| Diff. 4X naive (ZOH) | 2% (-24*) | 100% (+0) | 40% (-48*) | 98% (+0) | 85% (-4) |
| Diff. 4X +TAC-Fold+satfix | 2% (-24*) | 100% (+0) | 50% (-38*) | 98% (+0) | 84% (-5) |
| Diff. 4X +QP (ours) | 3% (-23*) | 100% (+0) | 52% (-36*) | 97% (-1) | 87% (-2) |

| FM configuration | PushT | Lift | PickCube | Can | Square |
|---|---|---|---|---|---|
| FM 1X Base | 30% | 100% | 89% | 100% | 84% |
| FM 2X naive (ZOH) | 11% (-19*) | 100% (+0) | 80% (-9*) | 100% (+0) | 86% (+2) |
| FM 2X +spline+satfix | 26% (-4) | 100% (+0) | 83% (-6) | 100% (+0) | 87% (+3) |
| FM 2X +TAC-Fold+satfix | 25% (-5) | 100% (+0) | 84% (-5) | 100% (+0) | 87% (+3) |
| FM 2X +QP (ours) | 23% (-7) | 100% (+0) | 84% (-5) | 100% (+0) | 84% (+0) |
| FM 4X naive (ZOH) | — | — | 40% (-49*) | — | — |
| FM 4X +TAC-Fold+satfix | — | — | 51% (-38*) | — | — |
| FM 4X +QP (ours) | — | — | 50% (-39*) | — | — |

cell = success% (delta vs 1X Base, paired exact McNemar; * p<0.05). n=100 except Diff. PickCube 4X n=400.
