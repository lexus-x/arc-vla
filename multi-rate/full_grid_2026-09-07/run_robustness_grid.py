#!/usr/bin/env python3
"""Run the empirical robustness evaluation grid for a specified task across K in {1, 2, 4, 8}."""
import argparse, subprocess, sys, time, os

CONDA_ENV = "vla_smolvla_libero"
HERE = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=HERE)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED with returncode {res.returncode} in {dt:.1f}s", flush=True)
        sys.exit(res.returncode)
    print(f"SUCCESS in {dt:.1f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["lift", "can", "square"])
    ap.add_argument("--obs_noise", type=float, default=0.05)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--ks", default="1,2,4,8")
    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",")]
    python_bin = "/home/user/anaconda3/envs/vla_smolvla_libero/bin/python"

    print(f"=== Starting Robustness Campaign for {args.task.upper()} (ks={ks}, noise={args.obs_noise}, n={args.n_eval}) ===", flush=True)
    for k in ks:
        cmd = [
            python_bin,
            f"{HERE}/harness_robustness.py",
            args.task,
            "--k", str(k),
            "--obs_noise", str(args.obs_noise),
            "--n_eval", str(args.n_eval),
        ]
        run_cmd(cmd)

    print(f"=== Finished Robustness Campaign for {args.task.upper()} ===", flush=True)

if __name__ == "__main__":
    main()
