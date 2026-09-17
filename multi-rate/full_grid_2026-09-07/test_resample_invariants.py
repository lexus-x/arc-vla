"""The one check this campaign was missing: every arm claims to redistribute a block sum,
and satfix additionally claims |a|<=1. Run: python test_resample_invariants.py"""
import numpy as np
from resample_math import RESAMPLERS, decimate_and_resample

K = 2
rng = np.random.default_rng(0)
bs = np.clip(rng.normal(0, 0.8, (20, 7)), -K, K)  # block sums are sums of k clipped actions

for name, fn in RESAMPLERS.items():
    out = fn(bs, K)
    err = np.abs(out.reshape(len(bs), K, bs.shape[1]).sum(1) - bs).max()
    sat = np.abs(out).max()
    print(f"{name:18s} blocksum_err={err:.2e}  max|a|={sat:.3f}")
    if name.endswith("satfix"):
        assert sat <= 1.0 + 1e-6, f"{name} violates |a|<=1: {sat}"
        assert err < 1e-5, f"{name} broke block-sum conservation: {err}"
    elif name == "bspline":
        # KNOWN BROKEN, kept as-is because the campaign is pre-registered: splprep is fit
        # parametrically and then evaluated at t_out as if t_out were the parameter, so the
        # cumulative curve is not reproduced and block sums are not conserved. This is why
        # raw bspline collapses (44% PickCube, 3% PushT) while bspline_satfix ~= zoh --
        # satfix's projection is repairing a bug, not smoothing. See FIRST_PRINCIPLES.md.
        assert err > 1e-2, "bspline unexpectedly conserves now -- was it fixed? update this test"
    else:
        assert err < 1e-5, f"{name} broke block-sum conservation: {err}"

# trailing remainder passes through untouched
x = rng.normal(0, 0.5, (9, 3))
assert np.allclose(decimate_and_resample(x, K, "zoh")[8:], x[8:])
assert len(decimate_and_resample(x, K, "zoh")) == 9
print("OK")
