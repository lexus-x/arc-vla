"""ctx arms: executed window keeps exact block sums + box; zoh_ctx == zoh (context can't matter)."""
import numpy as np, harness
from resample_math import coarsen_delta
rng = np.random.default_rng(0)
for k in (2, 4):
    harness.K = k
    pred = np.clip(rng.normal(0, .8, (16, 8)), -1, 1).astype(np.float32)
    prev = coarsen_delta(np.clip(rng.normal(0, .8, (8, 7)), -1, 1), k)
    for arm in ("qp_anchor", "tac_fold_satfix", "spline_satfix"):
        out = harness.apply_arm_ctx(pred, arm, 1, prev)
        assert out.shape == (8, 8) and np.abs(out).max() <= 1 + 1e-5, arm
        assert np.allclose(coarsen_delta(out[:, :7], k), coarsen_delta(pred[:8, :7], k), atol=1e-4), (arm, k)
        assert np.allclose(out[:, 7], harness.apply_arm(pred[:8], "zoh", 1)[:, 7])
    assert np.allclose(harness.apply_arm_ctx(pred, "zoh", 1, prev), harness.apply_arm(pred[:8], "zoh", 1))
print("ctx arm self-check ok")
