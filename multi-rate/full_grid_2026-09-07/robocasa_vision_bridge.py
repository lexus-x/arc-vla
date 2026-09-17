#!/usr/bin/env python3
"""Stage 1 RGB bridge. Run in RoboCasa's numpy-1.23 environment on localhost.

Pickle is a trusted-local protocol, matching the original bridge. This service
is separate so state-only clients and their cached datasets remain untouched.
"""
import argparse
import signal
import socket
import numpy as np
import sys

if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    for name in ("multiarray", "numeric", "_multiarray_umath", "umath"):
        mod = getattr(np.core, name, None)
        if mod is not None:
            sys.modules[f"numpy._core.{name}"] = mod

from robocasa_bridge import recv_msg, send_msg
from robocasa_vision_data import VisionDataset
from download_robocasa_vision_data import DEFAULT_OUTPUT_DIR


class VisionTaskState:
    def __init__(self, task, data_dir):
        self.data = VisionDataset(task, data_dir)
        self.env = None

    def request(self, msg):
        cmd = msg["cmd"]
        if cmd == "metadata":
            return {"metadata": self.data.metadata, "data_path": str(self.data.path)}
        if cmd == "harvest":
            return {"episodes": [self.data.episode(i) for i in msg["idx"]]}
        if cmd == "reset_to":
            if self.env is None:
                import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
                self.env = EnvUtils.create_env_for_data_processing(
                    env_meta=self.data.env_meta, reward_shaping=False, **self.data.camera_config())
            g = self.data.h5["data"][self.data.demos[msg["idx"]]]
            self.env.reset()
            od = self.env.reset_to({"states": np.asarray(g["states"][0]),
                                    "model": g.attrs["model_file"], "ep_meta": g.attrs.get("ep_meta")})
            return {"obs": self.data.live_observation(od)}
        if cmd == "step":
            if self.env is None:
                raise ValueError("reset_to must precede step")
            action = np.asarray(msg["action"], np.float32)
            if action.shape != (self.data.act_dim,) or not np.isfinite(action).all():
                raise ValueError("invalid native action")
            od, reward, done, info = self.env.step(action)
            return {"obs": self.data.live_observation(od),
                    "success": bool(self.env.is_success()["task"]), "done": bool(done)}
        raise ValueError(f"unknown vision command: {cmd}")

    def close(self):
        wrapper, self.env = self.env, None
        try:
            if wrapper is not None:
                wrapper.env.close()
        except BaseException as simulator_error:
            try:
                self.data.close()
            except BaseException as data_error:
                raise simulator_error from data_error
            raise
        else:
            self.data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    states = {}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", args.port))
            server.listen(1)
            print(f"[vision bridge] listening on 127.0.0.1:{args.port}", flush=True)
            while True:
                conn, _ = server.accept()
                with conn:
                    while True:
                        msg = recv_msg(conn)
                        if msg is None:
                            break
                        try:
                            if msg["cmd"] == "ping":
                                response = {"ok": True, "mode": "vision-stage1"}
                            elif msg["cmd"] == "close":
                                state = states.pop(msg["task"], None)
                                if state is not None:
                                    state.close()
                                response = {"ok": True}
                            else:
                                task = msg["task"]
                                if task not in states:
                                    states[task] = VisionTaskState(task, args.data_dir)
                                response = states[task].request(msg)
                        except Exception as exc:
                            response = {"error": f"{type(exc).__name__}: {exc}"}
                            print(f"[vision bridge] {response['error']}", flush=True)
                        send_msg(conn, response)
    except KeyboardInterrupt:
        pass
    finally:
        for state in states.values():
            state.close()


if __name__ == "__main__":
    main()
