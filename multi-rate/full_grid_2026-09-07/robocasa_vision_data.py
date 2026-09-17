"""Shared, torch-free contracts for Stage 1 HDF5 and live RGB observations."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from download_robocasa_vision_data import dataset_path, validate_dataset, TASK_URLS

PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
PAPER = {
    "TurnOffSinkFaucet": ("SinkFaucet", 79.0),
    "CoffeePressButton": ("CoffeePressButton", 93.0),
    "TurnOffMicrowave": ("Microwave", 77.0),
    "CloseSingleDoor": ("CloseDoor", 27.0),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VisionDataset:
    def __init__(self, task, data_dir):
        if task not in TASK_URLS:
            raise ValueError(f"unsupported Stage 1 task: {task}")
        self.path = dataset_path(Path(data_dir), task).resolve()
        summary = validate_dataset(self.path)
        self.h5 = h5py.File(self.path, "r")
        try:
            data = self.h5["data"]
            self.env_meta = json.loads(data.attrs["env_args"])
            recorded_task = self.env_meta.get("env_name")
            if recorded_task and recorded_task != task:
                raise ValueError(f"dataset task {recorded_task} does not match {task}")
            self.demos = sorted(data.keys(), key=lambda k: int(k.removeprefix("demo_")))
            self.camera_keys = tuple(summary.rgb_image_keys)
            if any(not k.endswith("_image") or "/" in k for k in self.camera_keys):
                raise ValueError("camera RGB keys must be flat <camera>_image names")
            first = data[self.demos[0]]
            self.widths = {k: int(first["obs"][k].shape[-1]) for k in PROPRIO_KEYS}
            if self.widths[PROPRIO_KEYS[0]] != 3 or self.widths[PROPRIO_KEYS[1]] != 4:
                raise ValueError("eef pose requires position(3) and quaternion(4)")
            self.shapes = {k: list(first["obs"][k].shape[1:]) for k in self.camera_keys}
            self.layouts = {}
            for k, shape in self.shapes.items():
                if len(shape) != 3 or (shape[-1] == 3) == (shape[0] == 3):
                    raise ValueError(f"ambiguous or invalid image shape: {k} {shape}")
                self.layouts[k] = "HWC" if shape[-1] == 3 else "CHW"
            self.act_dim = int(first["actions"].shape[-1])
            for name in self.demos:
                g = data[name]
                length = len(g["actions"])
                if length < 1 or g["actions"].shape != (length, self.act_dim):
                    raise ValueError(f"invalid actions in {name}")
                if "states" not in g or len(g["states"]) != length or "model_file" not in g.attrs:
                    raise ValueError(f"missing/alignment-invalid replay state in {name}")
                for k, width in self.widths.items():
                    if width < 1 or g["obs"][k].shape != (length, width):
                        raise ValueError(f"proprioception shape mismatch: {name}/{k}")
                for k, shape in self.shapes.items():
                    ds = g["obs"][k]
                    if list(ds.shape) != [length] + shape or ds.dtype != np.uint8:
                        raise ValueError(f"expected aligned raw uint8 RGB: {name}/{k}")
            self.metadata = {
                "task": task, "data_sha256": sha256_file(self.path),
                "demo_names": self.demos, "proprio_keys": list(PROPRIO_KEYS),
                "proprio_widths": self.widths, "camera_keys": list(self.camera_keys),
                "image_shapes": self.shapes, "image_layouts": self.layouts,
                "image_orientation": "robocasa_data_processing_wrapper_no_extra_flip",
                "act_dim": self.act_dim,
            }
        except Exception:
            self.h5.close()
            raise

    def close(self):
        self.h5.close()

    def episode(self, index):
        g = self.h5["data"][self.demos[index]]
        state = np.concatenate([np.asarray(g["obs"][k], np.float32) for k in PROPRIO_KEYS], -1)
        actions = np.asarray(g["actions"], np.float32)
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("nonfinite proprioception/actions")
        return {"state": state, "images": {k: np.asarray(g["obs"][k]) for k in self.camera_keys},
                "actions": actions}

    def live_observation(self, obs):
        parts = []
        for key in PROPRIO_KEYS:
            value = np.asarray(obs[key], np.float32).reshape(-1)
            if value.shape != (self.widths[key],) or not np.isfinite(value).all():
                raise ValueError(f"invalid live proprioception: {key}")
            parts.append(value)
        images = {}
        for key in self.camera_keys:
            # EnvUtils returns HWC uint8, already flipped identically to stored data.
            value = np.asarray(obs[key])
            if self.layouts[key] == "CHW":
                value = value.transpose(2, 0, 1)
            if list(value.shape) != self.shapes[key] or value.dtype != np.uint8:
                raise ValueError(f"invalid live RGB shape/dtype: {key}")
            images[key] = np.ascontiguousarray(value)
        return {"state": np.concatenate(parts), "images": images}

    def camera_config(self):
        sizes = []
        for key in self.camera_keys:
            shape = self.shapes[key]
            sizes.append(shape[:2] if self.layouts[key] == "HWC" else shape[1:])
        if any(size != sizes[0] for size in sizes):
            raise ValueError("bridge requires common camera resolution")
        return {"camera_names": [k[:-6] for k in self.camera_keys],
                "camera_height": sizes[0][0], "camera_width": sizes[0][1]}


def split_demos(total, n_train=35, n_eval=15):
    if n_train < 1 or n_eval < 1 or n_train + n_eval > total:
        raise ValueError(f"need disjoint positive train/eval counts, requested {n_train}+{n_eval} of {total}")
    return list(range(n_train)), list(range(n_train, n_train + n_eval))


class EpisodeWindows:
    """Keep each raw CPU frame once; materialize only the requested batch windows."""
    def __init__(self, episodes, state_norm, action_norm, n_obs=2, horizon=16):
        if not episodes or any(len(ep["actions"]) == 0 for ep in episodes):
            raise ValueError("nonempty episodes required")
        camera_keys = tuple(episodes[0]["images"])
        if not camera_keys or any(tuple(ep["images"]) != camera_keys for ep in episodes):
            raise ValueError("all episodes must have the same ordered camera keys")
        for ep in episodes:
            length = len(ep["actions"])
            if len(ep["state"]) != length or any(len(ep["images"][key]) != length for key in camera_keys):
                raise ValueError("state, image, and action trajectories must be aligned")
        self.episodes, self.state_norm, self.action_norm = episodes, state_norm, action_norm
        self.n_obs, self.horizon = n_obs, horizon
        self.ends = np.cumsum([len(ep["actions"]) for ep in episodes])

    def __len__(self):
        return int(self.ends[-1])

    def batch(self, indices):
        states, actions = [], []
        images = {k: [] for k in self.episodes[0]["images"]}
        for index in indices:
            if index < 0 or index >= len(self):
                raise IndexError(index)
            e = int(np.searchsorted(self.ends, index, side="right"))
            t = int(index - (self.ends[e - 1] if e else 0))
            ep = self.episodes[e]
            obs_idx = np.clip(np.arange(t - self.n_obs + 1, t + 1), 0, len(ep["actions"]) - 1)
            act_idx = np.clip(np.arange(t, t + self.horizon), 0, len(ep["actions"]) - 1)
            states.append(ep["state"][obs_idx])
            actions.append(ep["actions"][act_idx])
            for k in images:
                images[k].append(ep["images"][k][obs_idx])
        return (self.state_norm.norm(np.stack(states)), {k: np.stack(v) for k, v in images.items()},
                self.action_norm.norm(np.stack(actions)))
