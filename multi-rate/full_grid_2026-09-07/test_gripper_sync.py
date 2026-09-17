import subprocess
import sys

import numpy as np
import pytest

import resample_math
from eval_gripper_sync_openloop import evaluate_task
from resample_math import gripper_sync


def test_coarsen_gripper_transitions_records_start_and_value_of_final_run():
    raw = np.array(
        [-1, -1, 1, 1, 1, -1, -1, -1, -1],
        dtype=np.float32,
    )

    offsets, end_values = resample_math.coarsen_gripper_transitions(raw, k=4)

    np.testing.assert_array_equal(offsets, np.array([2, 1], dtype=np.int64))
    np.testing.assert_array_equal(end_values, np.array([1, -1], dtype=np.float32))
    assert np.issubdtype(offsets.dtype, np.integer)


def test_coarsen_gripper_transitions_uses_minus_one_for_unchanged_block():
    raw = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float32)

    offsets, end_values = resample_math.coarsen_gripper_transitions(raw, k=4)

    np.testing.assert_array_equal(offsets, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(end_values, np.array([-1, 1], dtype=np.float32))


def test_coarsen_gripper_transitions_ignores_transient_change_when_block_ends_at_start():
    raw = np.array([-1, 1, 1, -1], dtype=np.float32)

    offsets, end_values = resample_math.coarsen_gripper_transitions(raw, k=4)

    np.testing.assert_array_equal(offsets, np.array([-1], dtype=np.int64))
    np.testing.assert_array_equal(end_values, np.array([-1], dtype=np.float32))


def test_gripper_sync_reconstructs_from_block_summary_only_for_gripper():
    held = np.zeros((8, 4), dtype=np.float32)
    held[:, 2] = [-1, -1, -1, -1, 1, 1, 1, 1]
    held[:, 3] = [-1, -1, -1, -1, 1, 1, 1, 1]
    summary = (
        np.array([2, 1], dtype=np.int64),
        np.array([1, -1], dtype=np.float32),
    )

    synced = gripper_sync(held, summary, n_hold=2, k=4)

    np.testing.assert_array_equal(synced[:, 2], [-1, -1, 1, 1, 1, -1, -1, -1])
    np.testing.assert_array_equal(synced[:, 3], held[:, 3])


def test_gripper_sync_writes_recorded_nonbinary_end_value_without_inferring_negation():
    held = np.full((4, 1), 0.25, dtype=np.float32)
    summary = (
        np.array([1], dtype=np.int64),
        np.array([0.75], dtype=np.float32),
    )

    synced = gripper_sync(held, summary, n_hold=1, k=4)

    np.testing.assert_array_equal(synced[:, 0], [0.25, 0.75, 0.75, 0.75])


def test_gripper_sync_preserves_continuous_backbone_and_remainder():
    reconstructed = np.arange(30, dtype=np.float32).reshape(6, 5)
    reconstructed[:, :3] = -7
    reconstructed[:4, 3:] = [-1, 1]
    remainder = reconstructed[4:].copy()

    synced = gripper_sync(
        reconstructed,
        (np.array([3], dtype=np.int64), np.array([1], dtype=np.float32)),
        n_hold=2,
        k=4,
    )

    np.testing.assert_array_equal(synced[:, :3], reconstructed[:, :3])
    np.testing.assert_array_equal(synced[:4, 3], [-1, -1, -1, 1])
    np.testing.assert_array_equal(synced[:, 4], reconstructed[:, 4])
    np.testing.assert_array_equal(synced[4:], remainder)


def test_gripper_sync_rejects_wrong_number_of_block_summaries():
    chunk = np.zeros((8, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="one entry per complete block"):
        gripper_sync(
            chunk,
            (np.array([2]), np.array([1], dtype=np.float32)),
            n_hold=1,
            k=4,
        )


def test_openloop_evaluation_scores_summary_reconstruction_against_raw_target():
    actions = np.zeros((4, 2), dtype=np.float32)
    actions[:, 1] = [-1, -1, 1, 1]

    result = evaluate_task([actions], k=4, n_hold=1)

    assert result["gripper_steps"] == 32
    assert result["accuracy"]["zoh"] == 27 / 32
    assert result["accuracy"]["gripper_sync"] == 1.0


def test_harness_arm_defaults_to_spline_satfix_and_uses_coarsened_gripper_timing():
    # harness registers campaign-specific resamplers at import time. Keep those
    # mutations out of pytest's process so test_resample_invariants remains isolated.
    code = """
import numpy as np
import harness
raw = np.zeros((8, 3), dtype=np.float32)
raw[:, 0] = np.linspace(-0.5, 0.5, 8)
raw[:, 1] = np.linspace(0.25, -0.25, 8)
raw[:, 2] = [-1, -1, 1, 1, 1, -1, -1, -1]
harness.K = 4
synced = harness.apply_arm(raw, 'gripper_sync', n_hold=1)
spline = harness.apply_arm(raw, 'spline_satfix', n_hold=1)
np.testing.assert_allclose(synced[:, :2], spline[:, :2])
np.testing.assert_array_equal(synced[:, 2], raw[:, 2])
np.testing.assert_array_equal(spline[:, 2], [-1, -1, -1, -1, 1, 1, 1, 1])
"""
    subprocess.run([sys.executable, "-c", code], check=True)
