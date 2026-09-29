"""Compare sequential/parallel RoboCasa on two development seeds, not SR evidence.

Run from any Python; evaluations use the existing gr00t environment (NumPy 1.x
pickle compatibility with robocasa_uv is required).
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    root = Path(__file__).resolve().parent
    bridge_py = "/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
    eval_py = "/home/user/anaconda3/envs/gr00t/bin/python"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl")
    log_path = root / "parallel_rc_gr00t_probe.log"
    report = {"kind": "parallel_correctness_probe_not_SR", "runs": {}}
    active = None
    with log_path.open("w") as log:
        bridge = subprocess.Popen([bridge_py, "-u", "robocasa_parallel_bridge.py"], cwd=root,
                                  env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(100):
                lines = log_path.read_text().splitlines()
                if lines and lines[0].startswith("PORT="):
                    break
                if bridge.poll() is not None:
                    raise RuntimeError("bridge exited")
                time.sleep(.1)
            else:
                raise RuntimeError("bridge startup timeout")
            port = lines[0].split("=")[1]
            for workers in (1, 2):
                suffix = f"_gr00t_throughput_probe_20260928_w{workers}"
                result = root / f"result_dp_RC-CoffeePressButton_f0{suffix}.json"
                if result.exists():
                    raise FileExistsError(f"remove or archive previous probe before rerunning: {result}")
                cmd = [eval_py, "-u", "harness.py", "RC-CoffeePressButton", "--fold", "0",
                       "--n_train", "39", "--n_eval", "2", "--steps", "15000", "--k", "2",
                       "--arms", "native", "--port", port, "--rc-random-eval", "--eval-seed", "1000000",
                       "--workers", str(workers), "--suffix", suffix]
                start = time.perf_counter()
                with (root / f"rc_gr00t_throughput_w{workers}.log").open("w") as out:
                    active = subprocess.Popen(cmd, cwd=root, env=env, stdout=out,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                    code = active.wait(timeout=180)
                    if code:
                        raise RuntimeError(f"evaluation exited {code}")
                active = None
                data = json.loads(result.read_text())
                report["runs"][str(workers)] = {"wall_seconds": time.perf_counter() - start,
                    "success": data["success"], "saturation": data["policy_raw_sat_frac"],
                    "checkpoint_sha256": data["checkpoint_sha256"]}
                print(workers, report["runs"][str(workers)], flush=True)
            report["same_outputs"] = all(report["runs"]["1"][key] == report["runs"]["2"][key]
                                         for key in ("success", "saturation", "checkpoint_sha256"))
            assert report["same_outputs"], "parallel probe differs from sequential"
        except Exception as exc:
            report["error"] = str(exc)
            raise
        finally:
            (root / "parallel_rc_gr00t_probe.json").write_text(json.dumps(report, indent=2) + "\n")
            for proc in (active, bridge):
                if proc is not None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=10)


if __name__ == "__main__":
    main()
