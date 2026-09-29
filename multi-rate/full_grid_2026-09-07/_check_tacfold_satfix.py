"""Invariant check for tac_fold(+satfix): per-block sum preservation and box compliance."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from resample_math import coarsen_delta, decimate_and_resample

rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, (8, 4)).astype(np.float32)   # 8-step chunk, 4 dims, k=2
S = coarsen_delta(x, 2)

print("random in-box chunk (k=2)")
print(f"{'arm':16s} {'max|sum_err|':>12s} {'max|a|':>8s}  in-box")
for arm in ["zoh", "tac_fold", "tac_fold_satfix", "spline", "spline_satfix"]:
    out = decimate_and_resample(x, 2, arm)
    err = float(np.abs(out.reshape(len(S), 2, 4).sum(1) - S).max())
    mx = float(np.abs(out).max())
    print(f"{arm:16s} {err:12.2e} {mx:8.3f}  {mx <= 1 + 1e-6}")

# Adversarial: alternating saturated blocks -> steep Akima/cubic slopes -> intra-block overshoot
adv = np.array([[1.0] * 4, [1.0] * 4, [-1.0] * 4, [-1.0] * 4,
                [1.0] * 4, [1.0] * 4, [-1.0] * 4, [-1.0] * 4], dtype=np.float32)
S2 = coarsen_delta(adv, 2)
print("\nadversarial alternating-saturation chunk (k=2)")
for arm in ["tac_fold", "tac_fold_satfix", "spline", "spline_satfix"]:
    out = decimate_and_resample(adv, 2, arm)
    err = float(np.abs(out.reshape(len(S2), 2, 4).sum(1) - S2).max())
    mx = float(np.abs(out).max())
    print(f"{arm:16s} {err:12.2e} {mx:8.3f}  {mx <= 1 + 1e-6}")

# k=1 identity claim (harness.py:42 comment: "satfix at k=1 is identity")
out1 = decimate_and_resample(x, 1, "tac_fold_satfix")
print("\nk=1 satfix identity:", np.allclose(out1, x, atol=1e-6))
