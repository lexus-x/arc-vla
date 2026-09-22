"""
Evaluate TAC-Fold + satfix + QP (qp_anchor and tac_fold_satfix) on RoboMimic.
Rates:
  k=1 (20 Hz)
  k=2 (10 Hz)
  k=4 (5 Hz)
  k=8 (2.5 Hz)
"""
import subprocess, sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
python_bin = "/home/user/anaconda3/envs/vla_smolvla_libero/bin/python"

tasks = ["lift", "can", "square"]
ks = [2, 4, 8, 1]
arms = "qp_anchor,tac_fold_satfix"

for task in tasks:
    for k in ks:
        print(f"\n{'='*70}")
        print(f"RoboMimic: task={task}, k={k}, arms={arms}")
        print(f"{'='*70}", flush=True)

        cmd = [
            python_bin, "harness.py", task,
            "--arms", arms,
            "--k", str(k),
            "--n_eval", "50",
            "--workers", "4",
            "--suffix", "_qp_eval",
        ]
        t0 = time.time()
        res = subprocess.run(cmd, cwd=HERE)
        print(f"Finished {task} k={k} in {time.time()-t0:.1f}s (exit code {res.returncode})", flush=True)

print("\n" + "="*80)
print("ALL ROBOMIMIC QP EVALUATIONS COMPLETE!")
print("="*80, flush=True)
