"""
Same driver as run_robocasa_satfix_eval.py, restricted to the newly-added "spline" arm only
(zoh/tac_fold/tac_fold_satfix already have results from the prior run) -- same 2 tasks, same
n_episodes, for direct comparison against those numbers.
"""
import socket
import subprocess
import time

SCRATCH = "/tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad"
GR00T_PY = "/home/user/anaconda3/envs/gr00t/bin/python"
ROBOCASA_PY = "/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
CKPT = "/home/user/Isaac-GR00T/checkpoints/GR00T-N1.6-3B"
REPO = "/home/user/Isaac-GR00T"

ARMS = ["spline"]
TASKS = [
    "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
    "robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env",
]
N_EPISODES = 10
PORT = 5555


def wait_for_server(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                return True
        except OSError:
            time.sleep(2)
    return False


def main():
    for arm in ARMS:
        server_log = open(f"{SCRATCH}/robocasa_server_{arm}.log", "w")
        server = subprocess.Popen(
            [GR00T_PY, f"{SCRATCH}/run_gr00t_satfix_server.py",
             "--model-path", CKPT, "--embodiment-tag", "ROBOCASA_PANDA_OMRON",
             "--resampler", arm, "--decimation_k", "2"],
            stdout=server_log, stderr=subprocess.STDOUT, cwd=REPO,
        )
        print(f"[driver] arm={arm} server pid={server.pid} waiting for ready...", flush=True)
        if not wait_for_server():
            print(f"[driver] arm={arm} server FAILED to come up, skipping", flush=True)
            server.kill()
            continue
        time.sleep(5)

        for task in TASKS:
            task_short = task.split("/")[-1]
            out_log = f"{SCRATCH}/robocasa_eval_{arm}_{task_short}.log"
            print(f"[driver] arm={arm} task={task_short} n_episodes={N_EPISODES}", flush=True)
            with open(out_log, "w") as f:
                subprocess.run(
                    [ROBOCASA_PY, "gr00t/eval/rollout_policy.py",
                     "--n_episodes", str(N_EPISODES),
                     "--policy_client_host", "127.0.0.1",
                     "--policy_client_port", str(PORT),
                     "--env_name", task,
                     "--n_action_steps", "8",
                     "--n_envs", "1"],
                    stdout=f, stderr=subprocess.STDOUT, cwd=REPO,
                )
            print(f"[driver] arm={arm} task={task_short} DONE, see {out_log}", flush=True)

        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
        print(f"[driver] arm={arm} server stopped", flush=True)

    print("[driver] ALL ARMS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
