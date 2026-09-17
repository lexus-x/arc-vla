import io
import json
from pathlib import Path

import h5py
import pytest

import download_robocasa_vision_data as acquisition


def make_dataset(path: Path, *, camera: bool = True, camera_enabled: bool = True, demos: int = 2) -> None:
    with h5py.File(path, "w") as h5file:
        data = h5file.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_kwargs": {"use_camera_obs": camera_enabled}})
        for index in range(demos):
            obs = data.create_group(f"demo_{index}").create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=[[0.0, 0.0, 0.0]])
            if camera:
                obs.create_dataset("agentview_image", data=[[[[0, 0, 0]]]], dtype="uint8")


def test_validate_dataset_reports_camera_keys_and_demo_count(tmp_path: Path) -> None:
    path = tmp_path / "vision.hdf5"
    make_dataset(path)

    summary = acquisition.validate_dataset(path)

    assert summary.demo_count == 2
    assert summary.image_keys == ("agentview_image",)
    assert summary.rgb_image_keys == ("agentview_image",)


@pytest.mark.parametrize(
    ("camera", "camera_enabled", "message"),
    [
        (False, True, "no RGB image observations"),
        (True, False, "use_camera_obs"),
    ],
)
def test_validate_dataset_rejects_nonvision_files(
    tmp_path: Path, camera: bool, camera_enabled: bool, message: str
) -> None:
    path = tmp_path / "invalid.hdf5"
    make_dataset(path, camera=camera, camera_enabled=camera_enabled)

    with pytest.raises(acquisition.DatasetValidationError, match=message):
        acquisition.validate_dataset(path)


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int, content_range: str = "") -> None:
        self._body = io.BytesIO(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}

    def read(self, size: int) -> bytes:
        return self._body.read(size)

    def close(self) -> None:
        self._body.close()


def test_download_resumes_only_after_matching_range_response(tmp_path: Path) -> None:
    partial = tmp_path / "dataset.hdf5.partial"
    partial.write_bytes(b"first-")
    requests = []

    def opener(request, *, timeout: float):
        requests.append(request)
        return FakeResponse(b"second", status=206, content_range="bytes 6-11/12")

    acquisition.download_to_partial("https://example.invalid/file", partial, opener=opener)

    assert requests[0].get_header("User-agent") == "Mozilla/5.0"
    assert requests[0].get_header("Range") == "bytes=6-"
    assert partial.read_bytes() == b"first-second"


def test_download_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    partial = tmp_path / "dataset.hdf5.partial"
    partial.write_bytes(b"old-")

    def opener(request, *, timeout: float):
        return FakeResponse(b"replacement", status=200)

    acquisition.download_to_partial("https://example.invalid/file", partial, opener=opener)

    assert partial.read_bytes() == b"replacement"


def test_run_task_validates_partial_before_atomic_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.hdf5"
    make_dataset(source)

    def fake_download(url: str, partial_path: Path) -> None:
        partial_path.write_bytes(source.read_bytes())

    monkeypatch.setattr(acquisition, "download_to_partial", fake_download)
    summary = acquisition.run_task("CoffeePressButton", tmp_path, verify_only=False)
    final_path = acquisition.dataset_path(tmp_path, "CoffeePressButton")

    assert summary.path == final_path
    assert final_path.exists()
    assert not final_path.with_suffix(final_path.suffix + ".partial").exists()


def test_run_task_finalizes_a_valid_existing_partial_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = "TurnOffMicrowave"
    partial = acquisition.dataset_path(tmp_path, task).with_suffix(".hdf5.partial")
    make_dataset(partial)

    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("a valid partial must not be downloaded again")

    monkeypatch.setattr(acquisition, "download_to_partial", fail_if_called)
    summary = acquisition.run_task(task, tmp_path, verify_only=False)

    assert summary.demo_count == 2
    assert acquisition.dataset_path(tmp_path, task).exists()


def test_verify_mode_is_network_free_and_task_selectable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    task = "CoffeePressButton"
    make_dataset(acquisition.dataset_path(tmp_path, task))

    assert acquisition.main(["--verify", "--task", task, "--output-dir", str(tmp_path)]) == 0
    assert "demos: 2" in capsys.readouterr().out


def test_verify_mode_fails_for_missing_dataset(tmp_path: Path) -> None:
    assert acquisition.main(["--verify", "--task", "CloseSingleDoor", "--output-dir", str(tmp_path)]) == 1


def test_verify_mode_does_not_finalize_or_inspect_a_valid_partial(tmp_path: Path) -> None:
    task = "CloseSingleDoor"
    partial = acquisition.dataset_path(tmp_path, task).with_suffix(".hdf5.partial")
    make_dataset(partial)
    before = partial.read_bytes()

    assert acquisition.main(["--verify", "--task", task, "--output-dir", str(tmp_path)]) == 1
    assert partial.exists()
    assert partial.read_bytes() == before
    assert not acquisition.dataset_path(tmp_path, task).exists()


def test_refuses_state_only_directory_as_destination() -> None:
    with pytest.raises(ValueError, match="state-only"):
        acquisition._ensure_safe_output_dir(acquisition.STATE_ONLY_DATA_DIR)
