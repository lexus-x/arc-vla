"""Long-lived RoboCasa env server -- runs INSIDE the robocasa_uv venv (numpy 1.23-pinned;
robosuite/robocasa hard-assert this and can't share a process with the Blackwell-capable torch
the DP/FM policy needs). Exposes reset_to/step/harvest/close over a localhost length-prefixed
pickle socket so harness.py's RoboCasaSim (running in the `gr00t` conda env) can drive
closed-loop eval exactly like the in-process ManiSkillSim/RoboMimicSim adapters do.

Env creation and state replay reuse robocasa's own validated robomimic-style wrapper
(EnvUtils.create_env_for_data_processing + env.reset_to/get_observation/is_success), the same
code path robocasa's dataset_states_to_obs.py uses -- not a hand-rolled sim-state restore,
which is exactly the class of bug that silently desynced the ManiSkill PushT replay before
(physx_cuda vs physx_cpu backend mismatch, see multi-rate/satfix_2026-09-05 notes).

Usage: robocasa_uv/.venv/bin/python robocasa_bridge.py --port 8765
"""
import argparse, json, os, pickle, random, socket, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{HERE}/robocasa_data"

# Verified empirically (2026-09-07): RoboCasa PandaMobile action = 12 dims. dims[0:3]=rel_pos,
# dims[3:6]=rel_rot_axis_angle (both genuine continuous per-step deltas, saturate |a|>1 pre-clip
# like every other env here); dim[6]=gripper (binary); dims[7:11]=base motion (exactly 0 in every
# single-stage demo checked -- robot stays stationary); dim[11]=mode (constant -1). n_hold=6
# causal-holds dims 6:12 (gripper+base+mode); only dims 0:6 are resampled.
N_RESAMPLE_DIMS = 6


def recv_msg(conn):
    hdr = b""
    while len(hdr) < 4:
        chunk = conn.recv(4 - len(hdr))
        if not chunk: return None
        hdr += chunk
    (n,) = struct.unpack(">I", hdr)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk: raise ConnectionError("socket closed mid-message")
        buf += chunk
    return pickle.loads(buf)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(data)) + data)


class TaskState:
    """RoboCasa randomizes the scene's object/fixture set per episode, so the `object` obs key
    (and in principle any key) can have a DIFFERENT per-step width across episodes -- confirmed
    empirically 2026-09-07 (OpenDrawer demo_0 `object` is 56-dim, demo_1 is 28-dim, everything
    else fixed). Fix: scan every demo's obs shapes up front and zero-pad every key to its
    dataset-wide max width, both for harvested (stored) obs and for live reset_to/step obs --
    deterministic per (task, demo_idx), so the padding a policy sees in training exactly matches
    what it sees at eval for the same episode."""
    def __init__(self, env_task):
        import h5py
        import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

        self.h5 = h5py.File(f"{DATA_DIR}/{env_task}_ld.hdf5", "r")
        env_meta = json.loads(self.h5["data"].attrs["env_args"])
        self.env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta, camera_names=[], camera_height=84, camera_width=84, reward_shaping=False,
        )
        self.demos = sorted(self.h5["data"].keys(), key=lambda d: int(d[5:]))

    def obs_keys(self):
        if hasattr(self, "_keys"): return self._keys
        g0 = self.h5["data"][self.demos[0]]["obs"]
        keys = sorted(k for k in g0.keys() if np.asarray(g0[k]).ndim <= 2)  # drop any image obs
        widths = {k: 0 for k in keys}
        for d in self.demos:
            g = self.h5["data"][d]["obs"]
            for k in keys:
                widths[k] = max(widths[k], np.asarray(g[k]).shape[-1])
        self._keys, self._widths = keys, widths
        return self._keys

    def flat_obs(self, od, keys):
        parts = []
        for k in keys:
            v = np.asarray(od[k], np.float32).reshape(-1)
            w = self._widths[k]
            if v.shape[0] < w:
                v = np.concatenate([v, np.zeros(w - v.shape[0], np.float32)])
            parts.append(v)
        return np.concatenate(parts, 0)


def handle(conn, states):
    while True:
        msg = recv_msg(conn)
        if msg is None: return
        cmd = msg["cmd"]
        if cmd == "ping":
            send_msg(conn, {"ok": True}); continue
        task = msg["task"]
        if task not in states:
            states[task] = TaskState(task)
        st = states[task]
        keys = st.obs_keys()
        if cmd == "n_demos":
            send_msg(conn, {"n": len(st.demos)})
        elif cmd == "harvest":
            O, A = [], []
            for i in msg["idx"]:
                g = st.h5["data"][st.demos[i]]
                T = g["actions"].shape[0]
                obs_g = g["obs"]
                O.append(np.stack([st.flat_obs({k: obs_g[k][t] for k in keys}, keys) for t in range(T)]))
                A.append(np.asarray(g["actions"], np.float32))
            send_msg(conn, {"O": O, "A": A})
        elif cmd == "reset_to":
            g = st.h5["data"][st.demos[msg["idx"]]]
            state0 = {"states": np.asarray(g["states"][0]), "model": g.attrs["model_file"],
                      "ep_meta": g.attrs.get("ep_meta", None)}
            st.env.reset()
            od = st.env.reset_to(state0)
            send_msg(conn, {"obs": st.flat_obs(od, keys)})
        elif cmd == "reset_random":
            seed = msg["seed"]
            random.seed(seed)
            np.random.seed(seed)
            st.env.env.seed = seed
            st.env.env.rng = np.random.default_rng(seed)
            od = st.env.reset()
            send_msg(conn, {"obs": st.flat_obs(od, keys)})
        elif cmd == "step":
            od, r, done, info = st.env.step(msg["action"])
            send_msg(conn, {"obs": st.flat_obs(od, keys), "success": bool(info["is_success"]["task"]), "done": bool(done)})
        elif cmd == "close":
            # NOTE: per-client "close" must NOT tear down the shared TaskState -- states persist
            # across the whole campaign's sequential harness.py invocations (one bridge process,
            # many clients). Closing here previously killed the h5/env for every future client,
            # which is why fm_RC-OpenDrawer / dp_RC-PnPCounterToStove / fm_RC-PnPCounterToStove
            # all failed back-to-back right after dp_RC-OpenDrawer's clean exit (2026-09-07
            # 04:19 KST). Real teardown happens when this whole bridge process is killed at the
            # end of run_full_campaign.sh.
            send_msg(conn, {"ok": True})
        else:
            send_msg(conn, {"error": f"unknown cmd {cmd}"})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    states = {}
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port)); srv.listen(1)
    print(f"[bridge] listening on 127.0.0.1:{args.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[bridge] client connected {addr}", flush=True)
        try:
            handle(conn, states)
        except (ConnectionError, BrokenPipeError):
            print("[bridge] client disconnected", flush=True)
        except Exception as e:
            # One bad request must not kill the bridge for the rest of a multi-hour campaign --
            # log it and keep serving. (2026-09-07: an uncaught KeyError from a stale h5 handle
            # took the whole process down mid-campaign; that root cause is fixed above, but any
            # OTHER per-request exception should still degrade to "this client fails" not
            # "the bridge is gone".)
            print(f"[bridge] ERROR in handle(): {type(e).__name__}: {e}", flush=True)
        finally:
            try: conn.close()
            except Exception: pass


if __name__ == "__main__":
    main()
