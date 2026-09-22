"""
Evaluate qp_anchor on Push-T across rates:
  2.5Hz, 5.0Hz, 10.0Hz, 20.0Hz
"""
import subprocess, sys, os, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
python_bin = "/home/user/anaconda3/envs/vla_smolvla_libero/bin/python"

rates = ["2.5Hz", "5.0Hz", "10.0Hz", "20.0Hz"]
all_results = {}

for rate in rates:
    print(f"\n{'='*70}")
    print(f"Starting Push-T qp_anchor eval: rate={rate}")
    print(f"{'='*70}", flush=True)

    out_file = os.path.join(HERE, f"eval_pusht_qp_{rate}.json")
    cmd = [
        python_bin, "run_pusht_rate.py",
        "--rate", rate,
        "--arms", "qp_anchor",
        "--n_episodes", "50",
        "--batch_size", "10",
        "--start_seed", "1000",
        "--out_file", out_file,
    ]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=HERE)
    print(f"Finished {rate} in {time.time()-t0:.1f}s (exit code {res.returncode})", flush=True)
    if os.path.exists(out_file):
        with open(out_file) as f:
            all_results.update(json.load(f))

with open(os.path.join(HERE, "eval_pusht_qp_combined.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*80)
print("ALL PUSH-T QP EVALUATIONS COMPLETE!")
print("="*80, flush=True)
