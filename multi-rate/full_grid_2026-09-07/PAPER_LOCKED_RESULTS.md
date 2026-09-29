# Locked-claim closed-loop results

Every row is paired within one explicit n=400 result file. No globbing or largest-file selection is used.

| Task / rate | native | ZOH | TAC-Fold+satfix | QP | QP-anchor | QP-anchor vs TAC-Fold | QP-anchor vs QP |
|---|---:|---:|---:|---:|---:|---:|---:|
| PickCube-v1, 4X | 88.0% | 40.5% | 50.2% | 52.2% | 53.5% | +3.2 pp, p=0.0106 (18/5) | +1.2 pp, p=0.424 (15/10) |
| PushT-v1, 2X | 24.5% | 11.2% | 21.5% | 18.2% | 21.5% | +0.0 pp, p=1 (26/26) | +3.2 pp, p=0.171 (45/32) |

Discordant counts are shown as QP-anchor-only/reference-only. Exact two-sided McNemar p-values are uncorrected descriptive statistics; family-wise claims must use their preregistered Holm procedure.
