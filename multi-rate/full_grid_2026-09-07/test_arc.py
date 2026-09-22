import numpy as np
import pytest
import torch

import heads
from dp_min import DiffusionPolicy, VisualDiffusionPolicy
from stage1_vision import ARCVisualWindows, arc_action_normalizer


class IdentityNormalizer:
    @staticmethod
    def norm(value):
        return value


def test_arc_rate_condition_is_required_and_changes_the_network_input():
    policy = DiffusionPolicy(3, 2, horizon=8, rate_condition_dim=1,
                             down_dims=(16, 32, 64))
    obs = torch.zeros(2, 2, 3)
    act = torch.zeros(2, 8, 2)
    rates = torch.tensor([[0.0], [1.0]])

    assert torch.isfinite(policy.loss(obs, act, rates))
    with pytest.raises(ValueError, match="rate_condition"):
        policy.loss(obs, act)


@pytest.mark.parametrize("rate", heads.ARC_RATES)
def test_arc_targets_and_tacfold_preserve_predicted_block_displacement(rate):
    rng = np.random.default_rng(rate)
    actions = rng.uniform(-1, 1, (2, heads.ARC_BLOCKS * max(heads.ARC_RATES), 3))
    targets = heads.arc_targets(actions, nd=2, rate=rate)
    decoded = heads.decode(targets[0], "arc", nd=2, n=rate, arm="tac_fold")

    assert targets.shape == (2, heads.ARC_BLOCKS, 3)
    assert decoded.shape == (8, 3)
    complete_blocks = 8 // rate
    got = decoded[:, :2].reshape(complete_blocks, rate, 2).sum(1)
    expected = targets[0, :complete_blocks, :2] * rate
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_arc_rejects_untrained_rate():
    pred = np.zeros((heads.ARC_BLOCKS, 2), np.float32)
    with pytest.raises(ValueError, match="trained only"):
        heads.decode(pred, "arc", nd=2, n=3, arm="tac_fold")


@pytest.mark.parametrize("rate", [2, 4])
def test_arc_satfix_bounds_and_conserves_out_of_range_predictions(rate):
    pred = np.zeros((heads.ARC_BLOCKS, 3), np.float32)
    pred[:, 0], pred[:, 1] = 1.5, -2.0
    decoded = heads.decode(pred, "arc", nd=2, n=rate, arm="tac_fold_satfix")

    assert np.max(np.abs(decoded[:, :2])) <= 1 + 1e-6
    complete_blocks = len(decoded) // rate
    got = decoded[:, :2].reshape(complete_blocks, rate, 2).sum(1)
    expected = np.tile([rate, -rate], (complete_blocks, 1))
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_visual_policy_accepts_the_same_rate_condition():
    policy = VisualDiffusionPolicy(
        proprio_dim=3, act_dim=2, camera_keys=["front"], image_layouts={"front": "HWC"},
        backbone="tiny", embedding_dim=4, horizon=8, rate_condition_dim=1,
        down_dims=(16, 32, 64),
    )
    state = torch.zeros(2, 2, 3)
    images = {"front": torch.zeros(2, 2, 8, 8, 3, dtype=torch.uint8)}
    actions = torch.zeros(2, 8, 2)
    rates = torch.tensor([[0.0], [1.0]])

    assert torch.isfinite(policy.loss(state, images, actions, rates))
    assert policy.sample(state, images, n_steps=2, rate_condition=rates).shape == (2, 8, 2)


def test_visual_arc_windows_pair_each_image_window_with_all_trained_rates():
    action = np.arange(40 * 3, dtype=np.float32).reshape(40, 3) / 100
    episode = {"state": np.zeros((40, 2), np.float32),
               "images": {"front": np.zeros((40, 4, 4, 3), np.uint8)},
               "actions": action}
    normalizer = arc_action_normalizer([episode], n_hold=1)
    windows = ARCVisualWindows([episode], IdentityNormalizer(), normalizer, n_hold=1)
    states, images, targets, conditions = windows.batch([0, 40, 80])

    assert states.shape == (3, 2, 2) and images["front"].shape[:2] == (3, 2)
    np.testing.assert_array_equal(conditions[:, 0], [0.0, 0.5, 1.0])
    for row, rate in enumerate(heads.ARC_RATES):
        expected = heads.arc_targets(action[None], nd=2, rate=rate)[0]
        np.testing.assert_allclose(normalizer.denorm(targets[row]), expected, atol=1e-6)
