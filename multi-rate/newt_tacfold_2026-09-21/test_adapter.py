import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location("adapter", Path(__file__).with_name("eval_newt_tacfold.py"))
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def test_action_transforms():
    rng = np.random.default_rng(7)
    plan = rng.uniform(-1, 1, (8, 7)).astype(np.float32)
    for arm in adapter.ARMS:
        out = adapter.apply_arm(plan, arm)
        assert out.shape == plan.shape
        assert np.isfinite(out).all()
        if arm in {"zoh", "spline", "tac_fold"}:
            np.testing.assert_allclose(
                out[:, :6].reshape(4, 2, 6).sum(1),
                plan[:, :6].reshape(4, 2, 6).sum(1),
                atol=2e-6,
            )
        if arm != "native":
            np.testing.assert_array_equal(out[:, 6], np.repeat(plan[::2, 6], 2))

    odd = rng.uniform(-1, 1, (25, 7)).astype(np.float32)
    for arm in adapter.ARMS:
        out = adapter.apply_arm(odd, arm)
        assert out.shape == odd.shape
        if arm != "native":
            np.testing.assert_array_equal(out[:, 6], np.repeat(odd[::2, 6], 2)[:len(odd)])


if __name__ == "__main__":
    test_action_transforms()
    print("adapter checks passed")
