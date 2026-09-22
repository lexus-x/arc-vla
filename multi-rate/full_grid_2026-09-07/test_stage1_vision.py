import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from dp_min import DiffusionPolicy, MinMax, VisualDiffusionPolicy, prepare_rgb
from download_robocasa_vision_data import dataset_path
from robocasa_vision_data import EpisodeWindows, PAPER, PROPRIO_KEYS, VisionDataset, split_demos
from robocasa_vision_bridge import VisionTaskState
from stage1_vision import (
    DEFAULT_COMPARISON_ARMS,
    apply_comparison_arm,
    build_parser,
    campaign_artifact_status,
    comparison_result,
    decimated_comparison_result,
    decimated_rollout,
    experiment_request,
    load_checkpoint,
    native_rollout,
    save_checkpoint,
    validate_checkpoint,
    validate_result_request,
)


def tiny_architecture(camera_keys=("left_image", "wrist_image")):
    return {
        "proprio_dim": 9,
        "act_dim": 12,
        "camera_keys": list(camera_keys),
        "image_layouts": {key: "HWC" for key in camera_keys},
        "embedding_dim": 4,
        "backbone": "tiny",
        "n_obs": 2,
        "horizon": 16,
        "n_action_steps": 8,
        "T": 8,
        "down_dims": (8, 16),
    }


def checkpoint_metadata(camera_keys=("left_image", "wrist_image")):
    return {
        "task": "TurnOffSinkFaucet",
        "data_sha256": "synthetic-digest",
        "demo_names": [f"demo_{index}" for index in range(4)],
        "proprio_keys": list(PROPRIO_KEYS),
        "proprio_widths": {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2},
        "camera_keys": list(camera_keys),
        "image_shapes": {key: [8, 8, 3] for key in camera_keys},
        "image_layouts": {key: "HWC" for key in camera_keys},
        "image_orientation": "robocasa_data_processing_wrapper_no_extra_flip",
        "act_dim": 12,
    }


def valid_checkpoint(n_train=2, n_eval=2, *, steps=15_000, batch_size=256,
                     microbatch=16, seed=0):
    metadata = checkpoint_metadata()
    return {
        "format_version": 1,
        "ema_policy": {"sentinel": torch.tensor(1.0)},
        "architecture": tiny_architecture(),
        "data_metadata": metadata,
        "data_path": "/tmp/synthetic.hdf5",
        "state_normalizer": {
            "lo": np.zeros(9, np.float32), "hi": np.ones(9, np.float32), "rng": np.ones(9, np.float32)
        },
        "action_normalizer": {
            "lo": np.zeros(12, np.float32), "hi": np.ones(12, np.float32), "rng": np.ones(12, np.float32)
        },
        "train_demo_indices": list(range(n_train)),
        "eval_demo_indices": list(range(n_train, n_train + n_eval)),
        "budget": {
            "steps": steps, "batch_size": batch_size, "microbatch": microbatch, "lr": 1e-4,
            "seed": seed, "ddim_steps": 10, "ema_decay": 0.9999,
        },
    }


def test_visual_policy_tensor_contracts_and_state_policy_unchanged():
    policy = VisualDiffusionPolicy(**tiny_architecture())
    state = torch.zeros(2, 2, 9)
    images = {
        "left_image": np.zeros((2, 2, 8, 8, 3), dtype=np.uint8),
        "wrist_image": torch.full((2, 2, 8, 8, 3), 255.0),
    }
    action = torch.zeros(2, 16, 12)

    assert policy.encode(state, images).shape == (2, 2, 17)
    assert policy.loss(state, images, action).ndim == 0
    assert policy.sample(state, images, n_steps=2).shape == action.shape

    state_only = DiffusionPolicy(obs_dim=9, act_dim=12, T=8, down_dims=(8, 16))
    assert state_only.loss(state, action).ndim == 0


def test_production_backbone_is_untrained_resnet18_with_learned_projection():
    policy = VisualDiffusionPolicy(
        proprio_dim=9, act_dim=12, camera_keys=["camera_image"], embedding_dim=7,
        backbone="resnet18", n_obs=2, horizon=16, n_action_steps=8, T=8, down_dims=(8, 16),
    )

    assert policy.encoder.__class__.__name__ == "ResNet"
    assert isinstance(policy.encoder.fc, nn.Identity)
    assert isinstance(policy.projection, nn.Linear)
    assert policy.projection.out_features == 7
    encoded = policy.encode(
        torch.zeros(1, 2, 9),
        {"camera_image": np.zeros((1, 2, 32, 32, 3), np.uint8)},
    )
    assert encoded.shape == (1, 2, 16)


def test_rgb_preprocessing_preserves_orientation_and_handles_hwc_chw():
    hwc = np.zeros((1, 2, 3, 4, 3), dtype=np.uint8)
    hwc[..., 0, :, 0] = 255
    converted = prepare_rgb(hwc, "HWC")

    assert converted.shape == (1, 2, 3, 3, 4)
    assert converted.dtype == torch.float32
    assert torch.all(converted[..., 0, 0, :] == 1)
    assert torch.all(converted[..., 0, -1, :] == 0)
    chw = torch.from_numpy(hwc).movedim(-1, -3).float()
    assert torch.equal(prepare_rgb(chw, "CHW"), converted)
    with pytest.raises(ValueError, match=r"\[0,255\]"):
        prepare_rgb(np.full((2, 2, 3), 256.0, np.float32), "HWC")


class IdentityNormalizer:
    def norm(self, value):
        return value

    def denorm(self, value):
        return value


def test_multimodal_windows_keep_state_images_and_actions_aligned():
    values = np.array([10, 20, 30], np.float32)
    episode = {
        "state": values[:, None],
        "images": {
            "left_image": values.astype(np.uint8)[:, None, None, None],
            "wrist_image": (values + 1).astype(np.uint8)[:, None, None, None],
        },
        "actions": (values * 10)[:, None],
    }
    windows = EpisodeWindows([episode], IdentityNormalizer(), IdentityNormalizer(), n_obs=2, horizon=4)
    state, images, actions = windows.batch([0, 2])

    np.testing.assert_array_equal(state[..., 0], [[10, 10], [20, 30]])
    np.testing.assert_array_equal(images["left_image"][..., 0, 0, 0], [[10, 10], [20, 30]])
    np.testing.assert_array_equal(images["wrist_image"][..., 0, 0, 0], [[11, 11], [21, 31]])
    np.testing.assert_array_equal(actions[..., 0], [[100, 200, 300, 300], [300, 300, 300, 300]])


def test_multimodal_windows_reject_misaligned_trajectories():
    episode = {
        "state": np.zeros((2, 1)),
        "images": {"camera_image": np.zeros((1, 2, 2, 3), np.uint8)},
        "actions": np.zeros((2, 1)),
    }
    with pytest.raises(ValueError, match="aligned"):
        EpisodeWindows([episode], IdentityNormalizer(), IdentityNormalizer())


def make_vision_dataset(path: Path):
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        data.attrs["env_args"] = json.dumps({
            "env_name": "TurnOffSinkFaucet",
            "env_kwargs": {"use_camera_obs": True, "camera_names": ["derived_camera"]},
        })
        for demo_index in range(2):
            demo = data.create_group(f"demo_{demo_index}")
            demo.attrs["model_file"] = "<mujoco/>"
            demo.create_dataset("states", data=np.zeros((3, 5), np.float32))
            demo.create_dataset("actions", data=np.zeros((3, 12), np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=np.full((3, 3), 1 + demo_index, np.float32))
            obs.create_dataset("robot0_eef_quat", data=np.full((3, 4), 2 + demo_index, np.float32))
            obs.create_dataset("robot0_gripper_qpos", data=np.full((3, 2), 3 + demo_index, np.float32))
            obs.create_dataset("object", data=np.full((3, 99), 1000, np.float32))
            image = np.zeros((3, 4, 5, 3), np.uint8)
            image[:, 0, :, 0] = 255
            obs.create_dataset("derived_camera_image", data=image)


def test_dataset_uses_explicit_nonprivileged_state_and_derived_camera(tmp_path: Path):
    path = dataset_path(tmp_path, "TurnOffSinkFaucet")
    make_vision_dataset(path)
    data = VisionDataset("TurnOffSinkFaucet", tmp_path)
    try:
        episode = data.episode(0)
        assert PROPRIO_KEYS == ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
        assert "object" not in data.metadata["proprio_keys"]
        assert episode["state"].shape == (3, 9)
        assert np.max(episode["state"]) == 3
        assert data.camera_config() == {
            "camera_names": ["derived_camera"], "camera_height": 4, "camera_width": 5,
        }

        live_image = np.asarray(data.h5["data/demo_0/obs/derived_camera_image"][0])
        live = data.live_observation({
            "robot0_eef_pos": np.ones(3),
            "robot0_eef_quat": np.ones(4) * 2,
            "robot0_gripper_qpos": np.ones(2) * 3,
            "object": np.ones(200) * 1000,
            "derived_camera_image": live_image,
        })
        np.testing.assert_array_equal(live["images"]["derived_camera_image"], live_image)
        assert live["images"]["derived_camera_image"][0, 0, 0] == 255
        assert live["images"]["derived_camera_image"][-1, 0, 0] == 0
        assert live["state"].shape == (9,)
    finally:
        data.close()


def test_default_split_is_disjoint_35_train_15_eval():
    train_indices, eval_indices = split_demos(54)
    assert train_indices == list(range(35))
    assert eval_indices == list(range(35, 50))
    assert not set(train_indices) & set(eval_indices)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda checkpoint: checkpoint["data_metadata"].update(data_sha256="other"), "metadata mismatch"),
        (lambda checkpoint: checkpoint["architecture"].update(camera_keys=["other_image"]), "modalities"),
        (lambda checkpoint: checkpoint.update(eval_demo_indices=[1, 2]), "overlapping"),
        (lambda checkpoint: checkpoint["budget"].update(ddim_steps=9), "DDIM-10"),
        (lambda checkpoint: checkpoint["budget"].pop("seed"), "budget or seed"),
    ],
)
def test_checkpoint_metadata_validation_rejects_incompatibility(mutation, message):
    checkpoint = valid_checkpoint()
    expected_metadata = copy.deepcopy(checkpoint["data_metadata"])
    mutation(checkpoint)
    with pytest.raises(ValueError, match=message):
        validate_checkpoint(checkpoint, expected_metadata)


def test_checkpoint_roundtrip_contains_ema_architecture_normalizers_split_and_budget(tmp_path: Path):
    metadata = checkpoint_metadata()
    data = SimpleNamespace(metadata=metadata, path=tmp_path / "synthetic.hdf5")
    policy = VisualDiffusionPolicy(**tiny_architecture())
    state_norm = MinMax(np.stack([np.zeros(9), np.ones(9)]))
    action_norm = MinMax(np.stack([np.zeros(12), np.ones(12)]))
    path = tmp_path / "policy.pt"
    budget = valid_checkpoint()["budget"]

    save_checkpoint(path, policy, tiny_architecture(), data, state_norm, action_norm, [0, 1], [2, 3], budget)
    requested = experiment_request("TurnOffSinkFaucet", metadata, 2, 2, 15_000, 256, 16, 0, 0)
    loaded, loaded_state_norm, loaded_action_norm, checkpoint = load_checkpoint(
        path, metadata, "cpu", requested=requested)

    assert isinstance(loaded, VisualDiffusionPolicy)
    assert checkpoint["architecture"] == tiny_architecture()
    assert checkpoint["budget"]["seed"] == 0
    assert checkpoint["train_demo_indices"] == [0, 1]
    assert checkpoint["eval_demo_indices"] == [2, 3]
    np.testing.assert_array_equal(loaded_state_norm.lo, state_norm.lo)
    np.testing.assert_array_equal(loaded_action_norm.hi, action_norm.hi)
    incompatible_request = dict(requested, steps=1)
    with pytest.raises(ValueError, match="steps"):
        load_checkpoint(path, metadata, "cpu", requested=incompatible_request)


def write_result(path, checkpoint_path, data, checkpoint, episode_count, *, eval_seed=0, complete=True):
    episodes = [{"demo_index": index, "success": index % 2 == 0, "steps": 8}
                for index in range(episode_count)]
    result = comparison_result(
        data.metadata["task"], episodes, checkpoint_path, data, checkpoint, 500, eval_seed)
    result["complete"] = complete
    path.write_text(json.dumps(result))


def artifact_fixture(tmp_path, *, checkpoint=None, episode_count=None, eval_seed=0, complete=True):
    checkpoint = checkpoint or valid_checkpoint()
    checkpoint_path = tmp_path / "policy.pt"
    result_path = tmp_path / "result.json"
    data = SimpleNamespace(path=(tmp_path / "dataset.hdf5").resolve(),
                           metadata=checkpoint["data_metadata"])
    torch.save(checkpoint, checkpoint_path)
    if episode_count is not None:
        write_result(result_path, checkpoint_path, data, checkpoint, episode_count,
                     eval_seed=eval_seed, complete=complete)
    return checkpoint_path, result_path, data, checkpoint


def requested_experiment(data, *, n_train=2, n_eval=2, steps=15_000, batch_size=256,
                         microbatch=16, training_seed=0, eval_seed=0, max_steps=None):
    return experiment_request(
        data.metadata["task"], data.metadata, n_train, n_eval, steps, batch_size,
        microbatch, training_seed, eval_seed, max_steps)


def test_campaign_artifact_status_covers_fresh_eval_and_skip(tmp_path: Path):
    data = SimpleNamespace(path=(tmp_path / "dataset.hdf5").resolve(), metadata=checkpoint_metadata())
    checkpoint_path, result_path = tmp_path / "policy.pt", tmp_path / "result.json"
    requested = requested_experiment(data)
    assert campaign_artifact_status(checkpoint_path, result_path, data, requested) == "train-eval"

    checkpoint = valid_checkpoint()
    torch.save(checkpoint, checkpoint_path)
    assert campaign_artifact_status(checkpoint_path, result_path, data, requested) == "eval"

    write_result(result_path, checkpoint_path, data, checkpoint, 2)
    assert campaign_artifact_status(checkpoint_path, result_path, data, requested) == "skip"


def test_campaign_rejects_completed_smoke_for_production_request(tmp_path: Path):
    checkpoint_path, result_path, data, _ = artifact_fixture(
        tmp_path, checkpoint=valid_checkpoint(n_eval=1), episode_count=1)
    production = requested_experiment(data, n_eval=2)

    with pytest.raises(ValueError, match="evaluation split/count"):
        campaign_artifact_status(checkpoint_path, result_path, data, production)


def test_campaign_rejects_result_after_checkpoint_contents_change_at_same_path(tmp_path: Path):
    checkpoint_path, result_path, data, checkpoint = artifact_fixture(tmp_path, episode_count=2)
    checkpoint["ema_policy"]["sentinel"] = torch.tensor(2.0)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="checkpoint hash"):
        campaign_artifact_status(checkpoint_path, result_path, data, requested_experiment(data))


@pytest.mark.parametrize(
    "request_change, message",
    [
        ({"n_train": 1}, "training split/count"),
        ({"steps": 10}, "steps"),
        ({"batch_size": 8}, "batch size"),
        ({"microbatch": 4}, "microbatch"),
        ({"training_seed": 9}, "training seed"),
    ],
)
def test_campaign_rejects_checkpoint_budget_mismatch(tmp_path: Path, request_change, message):
    checkpoint_path, result_path, data, _ = artifact_fixture(tmp_path)
    requested = requested_experiment(data, **request_change)

    with pytest.raises(ValueError, match=message):
        campaign_artifact_status(checkpoint_path, result_path, data, requested)


def test_campaign_rejects_result_eval_seed_or_insufficient_complete_episodes(tmp_path: Path):
    checkpoint_path, result_path, data, checkpoint = artifact_fixture(
        tmp_path, episode_count=2, eval_seed=0)
    with pytest.raises(ValueError, match="evaluation seed"):
        campaign_artifact_status(
            checkpoint_path, result_path, data, requested_experiment(data, eval_seed=1))

    write_result(result_path, checkpoint_path, data, checkpoint, 1, complete=True)
    with pytest.raises(ValueError, match="at-least-requested episode count"):
        campaign_artifact_status(checkpoint_path, result_path, data, requested_experiment(data))


def test_campaign_rejects_short_horizon_result_for_default_request(tmp_path: Path):
    checkpoint_path, result_path, data, checkpoint = artifact_fixture(tmp_path)
    write_result(result_path, checkpoint_path, data, checkpoint, 2)
    result = json.loads(result_path.read_text())
    result["max_steps"] = 1
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="evaluation max steps"):
        campaign_artifact_status(checkpoint_path, result_path, data, requested_experiment(data))


def test_task_specific_max_steps_defaults_and_override():
    sink = checkpoint_metadata()
    coffee = dict(checkpoint_metadata(), task="CoffeePressButton")
    args = (2, 2, 15_000, 256, 16, 0, 0)

    assert experiment_request("TurnOffSinkFaucet", sink, *args)["max_steps"] == 500
    assert experiment_request("CoffeePressButton", coffee, *args)["max_steps"] == 300
    assert experiment_request("CoffeePressButton", coffee, *args, max_steps=17)["max_steps"] == 17


class CloseTracker:
    def __init__(self, failure=None):
        self.closed = False
        self.failure = failure

    def close(self):
        self.closed = True
        if self.failure is not None:
            raise self.failure


def test_vision_task_cleanup_closes_underlying_sim_and_hdf5_without_masking_failure():
    expected = RuntimeError("simulator close failed")
    simulator = CloseTracker(expected)
    data_failure = RuntimeError("HDF5 close failed")
    data = CloseTracker(data_failure)
    state = VisionTaskState.__new__(VisionTaskState)
    state.env = SimpleNamespace(env=simulator)
    state.data = data

    with pytest.raises(RuntimeError, match="simulator close failed") as raised:
        state.close()

    assert raised.value is expected
    assert raised.value.__cause__ is data_failure
    assert simulator.closed
    assert data.closed
    assert state.env is None


class FakePolicy:
    n_obs = 2
    n_action_steps = 8
    camera_keys = ("camera_image",)

    def sample(self, state, images, n_steps):
        assert state.shape == (1, 2, 1)
        assert images["camera_image"].shape == (1, 2, 2, 2, 3)
        assert n_steps == 10
        return torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)


class FakeClient:
    def __init__(self):
        self.actions = []
        self.observation = {"state": np.zeros(1, np.float32),
                            "images": {"camera_image": np.zeros((2, 2, 3), np.uint8)}}

    def request(self, command, **kwargs):
        if command == "reset_to":
            return {"obs": self.observation}
        assert command == "step"
        self.actions.append(np.asarray(kwargs["action"]))
        return {"obs": self.observation, "success": len(self.actions) == 8, "done": False}


def test_native_rollout_executes_first_eight_actions_directly_at_k1():
    client = FakeClient()
    result = native_rollout(client, FakePolicy(), IdentityNormalizer(), IdentityNormalizer(), 3, 100, "cpu")

    np.testing.assert_array_equal(np.stack(client.actions), np.arange(16).reshape(8, 2))
    assert result == {"demo_index": 3, "success": True, "steps": 8}


@pytest.mark.parametrize("arm", DEFAULT_COMPARISON_ARMS)
def test_comparison_arm_wiring_matches_harness_apply_arm_for_eight_step_chunk(arm):
    import harness

    chunk = np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8)
    chunk[:, 2] = [-1, -1, 1, 1, 1, -1, -1, -1]
    original_k = harness.K
    try:
        harness.K = 4
        expected = harness.apply_arm(chunk, arm, n_hold=6)
        actual = apply_comparison_arm(chunk, arm, n_hold=6, k=4)
    finally:
        harness.K = original_k

    assert actual.shape == (8, 8)
    np.testing.assert_array_equal(actual, expected)


class FakeComparisonPolicy:
    n_obs = 2
    n_action_steps = 8
    camera_keys = ("camera_image",)

    def sample(self, state, images, n_steps):
        assert state.shape == (1, 2, 1)
        assert images["camera_image"].shape == (1, 2, 2, 2, 3)
        assert n_steps == 10
        values = torch.linspace(-2, 2, 16 * 8, dtype=torch.float32)
        return values.reshape(1, 16, 8)


def test_decimated_rollout_clips_then_executes_one_transformed_eight_step_chunk():
    client = FakeClient()
    policy = FakeComparisonPolicy()

    result = decimated_rollout(
        client, policy, IdentityNormalizer(), IdentityNormalizer(), demo_index=3,
        ei=0, max_steps=100, dev="cpu", arm="zoh", k=4, n_hold=6)

    predicted = policy.sample(
        torch.zeros(1, 2, 1),
        {"camera_image": np.zeros((1, 2, 2, 2, 3), np.uint8)},
        n_steps=10,
    )[0].numpy()
    expected = apply_comparison_arm(
        np.clip(predicted[:8], -1, 1).astype(np.float32), "zoh", n_hold=6, k=4)
    np.testing.assert_array_equal(np.stack(client.actions), expected)
    assert result == {"demo_index": 3, "success": True, "steps": 8}


def test_eval_cli_parses_k_and_selected_comma_separated_arms():
    parser = build_parser()
    defaults = parser.parse_args(["CloseSingleDoor", "--mode", "eval"])
    selected = parser.parse_args([
        "CloseSingleDoor", "--mode", "eval", "--k", "4",
        "--arms", "native, zoh,gripper_sync",
    ])

    assert defaults.k == 2
    assert defaults.arms is None
    assert selected.k == 4
    assert selected.arms == ("native", "zoh", "gripper_sync")
    with pytest.raises(SystemExit):
        parser.parse_args(["CloseSingleDoor", "--arms", "native,unknown"])


def test_decimated_result_contains_paired_success_steps_and_exact_mcnemar(tmp_path: Path):
    import harness

    checkpoint_path = tmp_path / "policy.pt"
    checkpoint_path.write_bytes(b"checkpoint identity")
    checkpoint = valid_checkpoint()
    data = SimpleNamespace(path=tmp_path / "dataset.hdf5", metadata=checkpoint["data_metadata"])
    episodes = {
        "native": [
            {"demo_index": 2, "success": True, "steps": 8},
            {"demo_index": 3, "success": False, "steps": 16},
        ],
        "zoh": [
            {"demo_index": 2, "success": False, "steps": 24},
            {"demo_index": 3, "success": False, "steps": 32},
        ],
    }

    result = decimated_comparison_result(
        "CloseSingleDoor", episodes, checkpoint_path, data, checkpoint,
        max_steps=500, eval_seed=0, k=4, n_hold=6)

    assert result["success"] == {"native": [True, False], "zoh": [False, False]}
    assert result["step_counts"] == {"native": [8, 16], "zoh": [24, 32]}
    arm_only, reference_only, p_value = harness.exact_mcnemar(
        result["success"]["native"], result["success"]["zoh"])
    assert result["contrasts"]["native"]["vs_zoh"] == {
        "delta_pp": 50.0,
        "p": p_value,
        "arm_only": arm_only,
        "reference_only": reference_only,
    }


def test_decimated_result_validation_accepts_paired_arm_records_and_checks_rate(tmp_path: Path):
    checkpoint_path = tmp_path / "policy.pt"
    checkpoint = valid_checkpoint()
    checkpoint["data_metadata"]["task"] = "CloseSingleDoor"
    torch.save(checkpoint, checkpoint_path)
    data = SimpleNamespace(path=(tmp_path / "dataset.hdf5").resolve(),
                           metadata=checkpoint["data_metadata"])
    episodes = {
        "zoh": [{"demo_index": 2, "success": False, "steps": 8},
                {"demo_index": 3, "success": True, "steps": 8}],
        "tac_fold_satfix": [{"demo_index": 2, "success": True, "steps": 8},
                            {"demo_index": 3, "success": True, "steps": 8}],
    }
    result = decimated_comparison_result(
        "CloseSingleDoor", episodes, checkpoint_path, data, checkpoint,
        max_steps=500, eval_seed=0, k=4, n_hold=6)
    result["complete"] = True
    requested = experiment_request(
        "CloseSingleDoor", data.metadata, 2, 2, 15_000, 256, 16, 0, 0,
        head="step", k=4)

    validate_result_request(result, checkpoint_path, data, checkpoint, requested)
    with pytest.raises(ValueError, match="action resolution"):
        validate_result_request(result, checkpoint_path, data, checkpoint,
                                dict(requested, k=2))


@pytest.mark.parametrize("task", list(PAPER))
def test_paper_comparison_result_has_counts_mapping_gaps_and_identity(tmp_path: Path, task: str):
    checkpoint_path = tmp_path / "policy.pt"
    checkpoint_path.write_bytes(b"checkpoint identity")
    checkpoint = valid_checkpoint()
    data = SimpleNamespace(path=tmp_path / "dataset.hdf5", metadata=checkpoint["data_metadata"])
    episodes = [{"success": True}, {"success": False}, {"success": True}, {"success": False}]

    result = comparison_result(task, episodes, checkpoint_path, data, checkpoint, 500, 7)

    paper_task, reference = PAPER[task]
    assert result["task_mapping"] == {"dataset_task": task, "paper_task": paper_task}
    assert result["success_count"] == 2
    assert result["episode_count"] == 4
    assert result["success_percent"] == 50.0
    assert result["paper_diff_1x_base_percent"] == reference
    assert result["gap_percentage_points"] == 50.0 - reference
    assert result["checkpoint_sha256"]
    assert result["data_metadata"] == data.metadata
    assert "demo count is the next likely cause" in result["interpretation"]
    assert result["k"] == 1 and "no decimation/resampling" in result["action_execution"]
