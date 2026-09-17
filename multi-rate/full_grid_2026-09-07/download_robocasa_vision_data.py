#!/usr/bin/env python3
"""Download and validate the Stage 1 RoboCasa ``human_im`` datasets.

The downloader deliberately keeps its state in ``*.partial`` files.  A final
HDF5 path is created only after the file has passed the same validation used by
``--verify``.  This makes it safe to stop and resume a download without
mistaking a truncated file for a usable dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import h5py


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "robocasa_data_vision"
STATE_ONLY_DATA_DIR = ROOT / "robocasa_data"
USER_AGENT = "Mozilla/5.0"
TASK_URLS: dict[str, str] = {
    "TurnOffSinkFaucet": "https://utexas.box.com/shared/static/ceewfn4ydhprupdcdppfe8wu4x61oxdg.hdf5",
    "CoffeePressButton": "https://utexas.box.com/shared/static/l5dnmcfd0r36vhdqgjchxo20vajt7ohl.hdf5",
    "TurnOffMicrowave": "https://utexas.box.com/shared/static/0drm2h7fgd5857x8xgcj1lph23srpbj1.hdf5",
    "CloseSingleDoor": "https://utexas.box.com/shared/static/2wnm0u1x9fp9ni02pmzqzhpjsb1kgfrr.hdf5",
}


class DatasetValidationError(ValueError):
    """Raised when an HDF5 file is not a usable RoboCasa vision dataset."""


@dataclass(frozen=True)
class DatasetSummary:
    path: Path
    demo_count: int
    image_keys: tuple[str, ...]
    rgb_image_keys: tuple[str, ...]


def dataset_path(output_dir: Path, task: str) -> Path:
    """Return the stable, descriptive local file name for a task."""
    return output_dir / f"{task}_human_im.hdf5"


def _decode_env_args(value: object) -> dict[str, object]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    # numpy string scalars have an ``item`` method but importing numpy only for
    # this conversion would be unnecessary.
    if not isinstance(value, str) and hasattr(value, "item"):
        value = value.item()  # type: ignore[union-attr]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    if not isinstance(value, str):
        raise DatasetValidationError("data.attrs['env_args'] is not a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError("data.attrs['env_args'] is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise DatasetValidationError("data.attrs['env_args'] must decode to a JSON object")
    return parsed


def _iter_datasets(group: h5py.Group) -> Iterable[tuple[str, h5py.Dataset]]:
    found: list[tuple[str, h5py.Dataset]] = []

    def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            found.append((name, item))

    group.visititems(collect)
    return found


def _is_image_key(name: str) -> bool:
    lowered = name.lower()
    return "image" in lowered or "rgb" in lowered


def _is_rgb_observation(dataset: h5py.Dataset) -> bool:
    """Recognize common RoboMimic HWC and CHW RGB trajectory layouts."""
    shape = dataset.shape
    if len(shape) < 3:
        return False
    # Typical RoboMimic data is (time, height, width, channels).  The second
    # form supports channel-first data, (time, channels, height, width).
    return shape[-1] == 3 or (len(shape) >= 4 and shape[1] == 3)


def validate_dataset(path: Path) -> DatasetSummary:
    """Validate a completed human_im HDF5 and return its useful metadata."""
    try:
        with h5py.File(path, "r") as h5file:
            if "data" not in h5file or not isinstance(h5file["data"], h5py.Group):
                raise DatasetValidationError("missing top-level 'data' group")
            data = h5file["data"]
            if "env_args" not in data.attrs:
                raise DatasetValidationError("missing data.attrs['env_args']")
            env_args = _decode_env_args(data.attrs["env_args"])
            # RoboCasa records runtime settings inside ``env_kwargs``. Accept
            # the top-level form too, since it is a valid robomimic-style
            # serialization, while checking the actual setting rather than
            # merely the presence of a camera-like dataset.
            env_kwargs = env_args.get("env_kwargs", {})
            camera_enabled = env_args.get("use_camera_obs")
            if camera_enabled is None and isinstance(env_kwargs, dict):
                camera_enabled = env_kwargs.get("use_camera_obs")
            if camera_enabled is not True:
                raise DatasetValidationError("env_args must contain use_camera_obs: true")

            demo_names = list(data.keys())
            image_keys: set[str] = set()
            rgb_image_keys: set[str] = set()
            demos_without_rgb: list[str] = []
            for demo_name in demo_names:
                demo = data[demo_name]
                if not isinstance(demo, h5py.Group) or "obs" not in demo:
                    raise DatasetValidationError(f"demo {demo_name!r} has no obs group")
                obs = demo["obs"]
                if not isinstance(obs, h5py.Group):
                    raise DatasetValidationError(f"demo {demo_name!r} obs is not a group")
                demo_has_rgb = False
                for name, dataset in _iter_datasets(obs):
                    if _is_image_key(name):
                        image_keys.add(name)
                        if _is_rgb_observation(dataset):
                            rgb_image_keys.add(name)
                            demo_has_rgb = True
                if not demo_has_rgb:
                    demos_without_rgb.append(demo_name)

            if not rgb_image_keys:
                raise DatasetValidationError("no RGB image observations found under demo obs")
            if demos_without_rgb:
                preview = ", ".join(sorted(demos_without_rgb)[:3])
                raise DatasetValidationError(f"RGB image observations missing from demo(s): {preview}")
            return DatasetSummary(
                path=path,
                demo_count=len(demo_names),
                image_keys=tuple(sorted(image_keys)),
                rgb_image_keys=tuple(sorted(rgb_image_keys)),
            )
    except OSError as exc:
        raise DatasetValidationError(f"cannot open as HDF5: {exc}") from exc


def download_to_partial(
    url: str,
    partial_path: Path,
    *,
    timeout: float = 60.0,
    opener: Callable[..., object] = urlopen,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Fetch *url* into *partial_path*, appending only after a valid Range reply.

    ``urllib`` follows HTTP redirects by default.  Servers that ignore a Range
    request are handled by safely restarting the partial file rather than
    appending a second complete response to it.
    """
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    response = opener(request, timeout=timeout)
    try:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()  # type: ignore[union-attr]
        content_range = response.headers.get("Content-Range", "")  # type: ignore[union-attr]
        append = offset > 0 and status == 206 and content_range.startswith(f"bytes {offset}-")
        mode = "ab" if append else "wb"
        with partial_path.open(mode) as destination:
            while chunk := response.read(chunk_size):  # type: ignore[union-attr]
                destination.write(chunk)
    finally:
        response.close()  # type: ignore[union-attr]


def _print_summary(task: str, summary: DatasetSummary) -> None:
    print(f"[valid] {task}: {summary.path}")
    print(f"        demos: {summary.demo_count}")
    print(f"        camera image keys: {', '.join(summary.image_keys)}")
    print(f"        RGB image keys: {', '.join(summary.rgb_image_keys)}")


def _ensure_safe_output_dir(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_state_only = STATE_ONLY_DATA_DIR.resolve()
    if resolved_output == resolved_state_only or resolved_state_only in resolved_output.parents:
        raise ValueError(
            f"refusing output directory inside state-only data directory: {resolved_state_only}"
        )


def run_task(task: str, output_dir: Path, verify_only: bool) -> DatasetSummary:
    final_path = dataset_path(output_dir, task)
    partial_path = final_path.with_suffix(final_path.suffix + ".partial")
    if final_path.exists():
        return validate_dataset(final_path)
    if verify_only:
        raise FileNotFoundError(f"missing completed dataset: {final_path}")
    # A prior process may have finished receiving the bytes but stopped before
    # finalization. Avoid an unnecessary Range request (which some servers
    # answer with 416 for a complete local file) in that case.
    if partial_path.exists():
        try:
            summary = validate_dataset(partial_path)
        except DatasetValidationError:
            pass
        else:
            os.rename(partial_path, final_path)
            return DatasetSummary(final_path, summary.demo_count, summary.image_keys, summary.rgb_image_keys)

    print(f"[download] {task} -> {partial_path}", flush=True)
    download_to_partial(TASK_URLS[task], partial_path)
    summary = validate_dataset(partial_path)
    # ``rename`` is atomic within the output directory.  A pre-existing final
    # path was validated and returned above, so interrupted downloads can only
    # leave the resumable partial file behind.
    os.rename(partial_path, final_path)
    return DatasetSummary(final_path, summary.demo_count, summary.image_keys, summary.rgb_image_keys)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or verify RoboCasa Stage 1 human_im vision datasets."
    )
    parser.add_argument(
        "--task",
        choices=tuple(TASK_URLS),
        action="append",
        help="Task to process; may be repeated. Defaults to all four tasks.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validate completed local files only; never download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _ensure_safe_output_dir(args.output_dir)
        tasks = args.task or list(TASK_URLS)
        failures = 0
        for task in tasks:
            try:
                summary = run_task(task, args.output_dir, args.verify)
                _print_summary(task, summary)
            except (DatasetValidationError, FileNotFoundError, HTTPError, URLError, OSError) as exc:
                failures += 1
                print(f"[error] {task}: {exc}", file=sys.stderr)
        return 1 if failures else 0
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
